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
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    print(make(a.scene, a.n))
