"""Stand-in photo-tier input from a benchmark scene's video, following the capture protocol ("one photo
from each corner, looking across the room, so every wall is in at least two photos"):

- candidate frames: sharp frames (variance of the Laplacian) every ~0.25 s, excluding the first and
  last 8% (ARKitScenes captures start and end looking down at a floor marker);
- selection: N frames whose viewing directions (yaw) are spread evenly around the room, taken from the
  recorded trajectory. The poses only CHOOSE frames; they are never given to the pipeline, which sees
  plain upright JPEGs plus an intrinsics.json sidecar (what EXIF gives for real photos).

Video frames are softer than real stills (motion blur, compression), so photo-tier numbers from these
are pessimistic. The user's own benchmark has real per-room photos.

    python bench/make_photos.py <scene_dir> [--n 8]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from roomscope.frontends.video import _rotate_K, arkitscenes_K, find_video, upright_rotation


def _yaw_at(traj: np.ndarray, t: float) -> float | None:
    j = int(np.argmin(np.abs(traj[:, 0] - t)))
    if abs(traj[j, 0] - t) > 0.2:
        return None
    R = cv2.Rodrigues(traj[j, 1:4])[0].T          # camera -> world (ARKitScenes world is z-up)
    fwd = R[:, 2]                                  # OpenCV camera looks along +z
    if abs(fwd[2]) > 0.8:                          # looking at the floor or ceiling: no useful heading
        return None
    return float(np.arctan2(fwd[1], fwd[0]))


def make(scene: Path, n: int = 8, long_side: int = 1920) -> Path:
    out = Path("out/photo_inputs") / scene.name
    room = out / "room"
    if room.exists() and len(list(room.glob("*.jpg"))) == n:
        return out
    shutil.rmtree(out, ignore_errors=True)
    room.mkdir(parents=True)
    video = find_video(scene)
    traj = np.loadtxt(scene / "lowres_wide.traj")
    cap = cv2.VideoCapture(str(video))
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    W0, H0 = int(cap.get(3)), int(cap.get(4))
    rot = upright_rotation(video)
    t0 = traj[0, 0]                                 # .mov frame i ~ trajectory time t0 + i / fps
    cands = []
    for i in range(int(0.08 * total), int(0.92 * total), max(1, int(fps / 4))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            continue
        y = _yaw_at(traj, t0 + i / fps)
        if y is None:
            continue
        g = cv2.cvtColor(cv2.resize(f, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY)
        cands.append((i, y, cv2.Laplacian(g, cv2.CV_64F).var()))
    cands = np.array(cands)
    picks = []
    for b in range(n):                              # sharpest frame in each of n yaw sectors
        lo, hi = -np.pi + 2 * np.pi * b / n, -np.pi + 2 * np.pi * (b + 1) / n
        m = (cands[:, 1] >= lo) & (cands[:, 1] < hi)
        if m.any():
            picks.append(int(cands[m][np.argmax(cands[m, 2]), 0]))
    K0 = arkitscenes_K(video, W0, H0)
    Ks = {}
    for k, i in enumerate(sorted(picks)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        K = _rotate_K(K0, rot, W0, H0) if K0 is not None else None
        if rot is not None:
            f = cv2.rotate(f, rot)
        h, w = f.shape[:2]
        sc = long_side / max(h, w)
        if sc < 1:
            f = cv2.resize(f, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
            if K is not None:
                K = K.copy(); K[:2] *= sc
        p = room / f"photo_{k:02d}.jpg"
        cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if K is not None:
            Ks[p.name] = K.round(3).tolist()
    cap.release()
    if Ks:
        (room / "intrinsics.json").write_text(json.dumps(Ks, indent=1))
    # provenance for evaluation only (out/, never read by the pipeline): source frame and trajectory time
    (out / "sources.json").write_text(json.dumps({f"photo_{k:02d}.jpg": {"frame": i, "t": t0 + i / fps}
                                                  for k, i in enumerate(sorted(picks))}, indent=1))
    return out


def _turn_K(K: np.ndarray, k: int, w: int, h: int) -> np.ndarray:
    """Intrinsics after np.rot90(image, k) (k counter-clockwise quarter turns) of a w x h image."""
    K = K.copy()
    for _ in range(k % 4):                      # one CCW turn: (u, v) -> (v, w - 1 - u)
        K = np.array([[K[1, 1], 0, K[1, 2]], [0, K[0, 0], w - K[0, 2]], [0, 0, 1.0]])
        w, h = h, w
    return K


def make_stray(capture: Path, n: int = 8, select: Path | None = None, long_side: int = 1920) -> Path:
    """Photo-tier input from a Stray recording (MuSHRoom conversions, own iPhone recordings).

    Without `select`: one folder, the sharpest frame in each of n heading sectors over the whole recording.
    With `select` (benchmark/photo_select/<id>.yaml: room name -> seconds into rgb.mp4, chosen by eye): one folder
    per room with exactly those frames. Picking rooms automatically from the LiDAR partition was tried and gave
    doorway shots of the next room and missed the office desk, so multi-room selection is by hand."""
    import yaml
    from roomscope.core.semantics import upright_k
    from roomscope.frontends.lidar import ARKIT_TO_ZUP, CV_TO_OURS, _read_odometry
    from scipy.spatial.transform import Rotation

    capture = Path(capture)
    out = Path("out/photo_inputs") / (capture.name + ("_rooms" if select else ""))
    shutil.rmtree(out, ignore_errors=True)
    odo = _read_odometry(capture / "odometry.csv")
    poses = []
    for j in range(len(odo["frame"])):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([odo[q][j] for q in ("qx", "qy", "qz", "qw")]).as_matrix()
        T[:3, 3] = [odo["x"][j], odo["y"][j], odo["z"][j]]
        poses.append(ARKIT_TO_ZUP @ T @ CV_TO_OURS)              # camera -> z-up world, camera looks along -z
    poses = np.array(poses)
    t = odo["timestamp"] - odo["timestamp"][0]
    cap = cv2.VideoCapture(str(capture / "rgb.mp4"))
    W0, H0, fps = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS)
    if select:
        groups = {name: [int(np.argmin(np.abs(t - s))) for s in secs]
                  for name, secs in yaml.safe_load(Path(select).read_text()).items()}
    else:
        span = np.where((t > 0.03 * t[-1]) & (t < 0.97 * t[-1]))[0]
        cands = []
        for j in span[::max(1, int(len(t) / t[-1] / 4))]:
            fwd = -poses[j][:3, 2]
            if abs(fwd[2]) > 0.8:                                 # looking at the floor or ceiling
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(odo["frame"][j]))
            ok, f = cap.read()
            if ok:
                g = cv2.cvtColor(cv2.resize(f, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY)
                cands.append((j, np.arctan2(fwd[1], fwd[0]), cv2.Laplacian(g, cv2.CV_64F).var()))
        cands = np.array(cands)
        picks = []
        for b in range(n):
            lo, hi = -np.pi + 2 * np.pi * b / n, -np.pi + 2 * np.pi * (b + 1) / n
            m = (cands[:, 1] >= lo) & (cands[:, 1] < hi)
            if m.any():
                picks.append(int(cands[m][np.argmax(cands[m, 2]), 0]))
        groups = {"room": sorted(picks)}
    sources = {}
    for name, rows in groups.items():
        room = out / name
        room.mkdir(parents=True)
        Ks, thumbs = {}, []
        for k, j in enumerate(rows):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(odo["frame"][j]))
            ok, f = cap.read()
            if not ok:
                continue
            K = np.array([[odo["fx"][j], 0, odo["cx"][j]], [0, odo["fy"][j], odo["cy"][j]], [0, 0, 1.0]])
            r = upright_k(poses[j])
            f = np.ascontiguousarray(np.rot90(f, r))
            K = _turn_K(K, r, W0, H0)
            h, w = f.shape[:2]
            sc = long_side / max(h, w)
            if sc < 1:
                f = cv2.resize(f, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
                K[:2] *= sc
            p = room / f"photo_{k:02d}.jpg"
            cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
            Ks[p.name] = K.round(3).tolist()
            sources[f"{name}/{p.name}"] = {"frame": int(odo["frame"][j]), "t": float(t[j])}
            thumbs.append(cv2.resize(f, (240, int(240 * f.shape[0] / f.shape[1]))))
        (room / "intrinsics.json").write_text(json.dumps(Ks, indent=1))
        hmax = max(x.shape[0] for x in thumbs)
        cv2.imwrite(str(out / f"contact_{name}.jpg"),
                    np.hstack([np.pad(x, ((0, hmax - x.shape[0]), (0, 0), (0, 0))) for x in thumbs]))
        print(f"  {name}: {len(Ks)} photos")
    cap.release()
    (out / "sources.json").write_text(json.dumps(sources, indent=1))    # provenance, never read by the pipeline
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--select", type=Path, default=None, help="Stray only: per-room seconds (benchmark/photo_select)")
    a = ap.parse_args()
    if (a.scene / "odometry.csv").exists():
        print(make_stray(a.scene, a.n, a.select))
    else:
        print(make(a.scene, a.n))
