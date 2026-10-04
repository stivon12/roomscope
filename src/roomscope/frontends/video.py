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

from .recon import run_mapanything, to_capture

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


def extract_keyframes(video: Path, out_dir: Path, n: int = 32, long_side: int = 1024,
                      trim: float = 0.0) -> tuple[list[Path], np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    rot = upright_rotation(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    bins = np.linspace(int(trim * total), int((1 - trim) * total), n + 1).astype(int)
    step = max(1, int(fps / 6))                 # score ~6 candidates per second within each bin
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, ts, Ks = [], [], {}
    W0, H0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K0 = arkitscenes_K(video, W0, H0)
    for b in range(n):
        best, best_s, best_i = None, -1.0, -1
        for i in range(bins[b], bins[b + 1], step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(cv2.resize(f, None, fx=0.25, fy=0.25), cv2.COLOR_BGR2GRAY)
            s = cv2.Laplacian(g, cv2.CV_64F).var()
            if s > best_s:
                best, best_s, best_i = f, s, i
        if best is None:
            continue
        K = None if K0 is None else _rotate_K(K0, rot, W0, H0)
        if rot is not None:
            best = cv2.rotate(best, rot)
        h, w = best.shape[:2]
        sc = long_side / max(h, w)
        if sc < 1:
            best = cv2.resize(best, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
            if K is not None:
                K = K.copy(); K[:2] *= best.shape[1] / w
        p = out_dir / f"kf_{b:03d}_{best_i:06d}.jpg"
        if K is not None:
            Ks[p.name] = K.round(3).tolist()
        cv2.imwrite(str(p), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(p)
        ts.append(best_i / fps)
    cap.release()
    if Ks:
        import json
        (out_dir / "intrinsics.json").write_text(json.dumps(Ks, indent=1))
    return paths, np.asarray(ts)


def load_video(path: Path, work_dir: Path, n_frames: int = 32, scale: float = 1.0):
    video = find_video(path)
    paths, ts = extract_keyframes(video, Path(work_dir) / "frames", n=n_frames)
    views = run_mapanything(paths, cache=Path(work_dir) / "cache")
    return to_capture(views, ts, Path(path), scale=scale)
