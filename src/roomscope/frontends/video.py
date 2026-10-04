"""Video tier: a handheld walkthrough (any phone camera app) -> metric posed points.

1. Keyframes: the video is cut into N equal time bins; in each bin the sharpest frame (variance of
   the Laplacian, which rejects motion blur) is kept, so coverage follows the walk evenly.
2. MapAnything reconstructs all keyframes jointly (frontends/recon.py), in metric units.
3. The result is a LidarCapture, so the shared core runs unchanged.

Frames are written to out/<capture>/frames/ so a run can be inspected and replayed.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .recon import (clean_sfm_track, reray_known_K, run_mapanything, run_mapanything_posed, sfm_poses,
                    to_capture)

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi"}


def find_video(path: Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    vids = sorted(p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXT)
    if not vids:
        raise FileNotFoundError(f"no video file under {path}")
    return vids[0]


ROTATE = {"Up": None, "Left": cv2.ROTATE_90_CLOCKWISE, "Right": cv2.ROTATE_90_COUNTERCLOCKWISE, "Down": cv2.ROTATE_180}


def upright_rotation(path: Path):
    """ARKitScenes stores frames in sensor orientation and records where the sky is (metadata.csv
    sky_direction). Phone camera apps store their own orientation flag, which OpenCV applies itself, so
    for ordinary captures this returns None."""
    path = Path(path)
    vid = path.stem if path.is_file() else path.name
    for parent in [path] + list(path.parents)[:4]:
        md = parent / "metadata.csv"
        if md.exists():
            import csv
            for row in csv.DictReader(md.open()):
                if row.get("video_id") == vid:
                    return ROTATE.get(row.get("sky_direction", "Up"))
    return None


def arkitscenes_K(video: Path, W: int, H: int) -> np.ndarray | None:
    """ARKitScenes: the .mov is the wide camera at full resolution; lowres_wide_intrinsics/*.pincam hold
    the same camera at 256x192. Returns K for a W x H sensor-orientation frame, or None elsewhere."""
    pins = sorted((Path(video).parent / "lowres_wide_intrinsics").glob("*.pincam"))
    if not pins:
        return None
    w, h, fx, fy, cx, cy = np.median(np.array([np.loadtxt(p) for p in pins[::50]]), 0)
    s = W / w
    return np.array([[fx * s, 0, cx * s], [0, fy * s * H / (h * s), cy * H / h], [0, 0, 1.0]])


def _rotate_K(K: np.ndarray, rot, W: int, H: int) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if rot == cv2.ROTATE_90_CLOCKWISE:          # (u, v) -> (H-1-v, u)
        return np.array([[fy, 0, H - 1 - cy], [0, fx, cx], [0, 0, 1.0]])
    if rot == cv2.ROTATE_90_COUNTERCLOCKWISE:   # (u, v) -> (v, W-1-u)
        return np.array([[fy, 0, cy], [0, fx, W - 1 - cx], [0, 0, 1.0]])
    if rot == cv2.ROTATE_180:
        return np.array([[fx, 0, W - 1 - cx], [0, fy, H - 1 - cy], [0, 0, 1.0]])
    return K


def frame_times(video: Path, total: int, fps: float) -> np.ndarray:
    """Capture time of every .mov frame. ARKitScenes ships one lowres_wide_intrinsics/*.pincam per video
    frame, named by its timestamp; their count equals the frame count, and lowres_wide.traj starts later
    (0.633 s on 42444946), so `traj_start + i/fps` is wrong. Elsewhere: i / fps."""
    pins = sorted((Path(video).parent / "lowres_wide_intrinsics").glob("*.pincam"))
    if len(pins) == total:
        return np.array([float(p.stem.split("_")[-1]) for p in pins])
    return np.arange(total) / fps


def extract_keyframes(video: Path, out_dir: Path, n: int = 32, long_side: int = 1024,
                      trim: float = 0.0) -> tuple[list[Path], np.ndarray]:
    """n keyframes: the sharpest frame (variance of the Laplacian) in each of n equal time bins, scoring
    ~6 candidates per second. One sequential decoding pass (grab, retrieve only candidates): seeking
    before every candidate re-decoded from the previous keyframe and took ~205 s of a 392 s run."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    rot = upright_rotation(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    times = frame_times(video, total, fps)
    bins = np.linspace(int(trim * total), int((1 - trim) * total), n + 1).astype(int)
    step = max(1, int(fps / 6))
    out_dir.mkdir(parents=True, exist_ok=True)
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # the ARKitScenes sensor is landscape. Some .mov files carry a rotation flag that OpenCV applies
    # itself (41069042 decodes as 1440x1920), others do not (42444946: 1920x1440). If the decoder already
    # rotated, rotating again would turn the frame (and its K) 90 deg too far
    auto_rotated = rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE) and H0 > W0
    Ws, Hs = (H0, W0) if auto_rotated else (W0, H0)          # sensor-orientation size
    K0 = arkitscenes_K(video, Ws, Hs)
    best = [(-1.0, -1, None)] * n                            # (score, frame index, image) per bin
    b = 0
    for i in range(total):
        if not cap.grab():
            break
        while b < n and i >= bins[b + 1]:
            b += 1
        if b >= n:
            break
        if i < bins[b] or (i - bins[b]) % step:
            continue
        ok, f = cap.retrieve()
        if not ok:
            continue
        g = cv2.cvtColor(cv2.resize(f, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY)
        sc = cv2.Laplacian(g, cv2.CV_64F).var()
        if sc > best[b][0]:
            best[b] = (sc, i, f)
    cap.release()
    paths, ts, Ks = [], [], {}
    for k, (_, i, img) in enumerate(best):
        if img is None:
            continue
        K = None if K0 is None else _rotate_K(K0, rot, Ws, Hs)
        if rot is not None and not auto_rotated:
            img = cv2.rotate(img, rot)
        h, w = img.shape[:2]
        sc = long_side / max(h, w)
        if sc < 1:
            img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
            if K is not None:
                K = K.copy(); K[:2] *= img.shape[1] / w
        p = out_dir / f"kf_{k:03d}_{i:06d}.jpg"
        if K is not None:
            Ks[p.name] = K.round(3).tolist()
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(p)
        ts.append(times[i])
    if Ks:
        import json
        (out_dir / "intrinsics.json").write_text(json.dumps(Ks, indent=1))
    return paths, np.asarray(ts)


def load_video(path: Path, work_dir: Path, n_frames: int = 32, scale: float = 1.0, method: str = "sfm",
               n_dense: int = 200):
    """method="sfm" (default):
    1. n_dense keyframes (sharpest per time bin, one decoding pass);
    2. COLMAP (subprocess): SIFT, exhaustive matching (loop closures), incremental mapping, known K fixed;
    3. drop frames that break the camera path's continuity (clean_sfm_track);
    4. MapAnything on n_frames of the registered frames, conditioned on COLMAP's intrinsics and poses:
       metric depth and metric poses (run_mapanything_posed).
    On 42444946 step 2 registered 145/200 frames; 112 within 1.9 cm (median) of ARKit after a similarity
    fit. Falls back to method="mapanything" (MapAnything alone on n_frames keyframes) when fewer than 8
    frames register."""
    import json
    video = find_video(path)
    work_dir = Path(work_dir)
    if method == "sfm":
        dense, ts_d = extract_keyframes(video, work_dir / "frames_dense", n=n_dense, long_side=1024)
        kp = work_dir / "frames_dense" / "intrinsics.json"
        Ks = json.loads(kp.read_text()) if kp.exists() else {}
        K = np.asarray(Ks[dense[0].name]) if dense and dense[0].name in Ks else None
        if K is None:
            from .recon import image_intrinsics
            K = image_intrinsics(dense[0]) if dense else None
        sfm = sfm_poses(dense, work_dir / "sfm", K)
        times = {p.name: t for p, t in zip(dense, ts_d)}
        sfm, dropped = clean_sfm_track(sfm, times)
        reg = [i for i, p in enumerate(dense) if p.name in sfm]
        load_video.diag = {"dense": len(dense), "registered": len(reg) + len(dropped), "dropped_jumps": len(dropped)}
        print(f"SfM: {len(reg) + len(dropped)}/{len(dense)} registered, {len(dropped)} dropped as path jumps")
        if len(reg) >= 8 and K is not None:
            pick = [reg[int(round(j))] for j in np.linspace(0, len(reg) - 1, min(n_frames, len(reg)))]
            paths = [dense[i] for i in pick]
            views, s_metric = run_mapanything_posed(paths, K, [sfm[p.name]["pose"] for p in paths],
                                                    cache=work_dir / "cache")
            load_video.diag["metric_scale"] = s_metric
            # evaluation provenance (out/, never read back by the pipeline): which frames, metric camera centres
            (work_dir / "video_diag.json").write_text(json.dumps({
                **load_video.diag, "dropped": dropped,
                "picked": {p.name: [float(x) for x in v["pose"][:3, 3]] for p, v in zip(paths, views)},
                "frame_times": {p.name: float(t) for p, t in zip(dense, ts_d)}}, indent=1))
            return to_capture(views, ts_d[pick], Path(path), scale=scale, snap=False)
        print("SfM too sparse or no intrinsics; falling back to MapAnything alone")
    paths, ts = extract_keyframes(video, work_dir / "frames", n=n_frames)
    views = run_mapanything(paths, cache=work_dir / "cache")
    reray_known_K(views, paths)
    return to_capture(views, ts, Path(path), scale=scale)
