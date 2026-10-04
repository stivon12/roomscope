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

from .recon import fuse_sfm_depth, run_mapanything, sfm_poses, to_capture

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
    # the ARKitScenes sensor is landscape. Some .mov files carry a rotation flag that OpenCV applies
    # itself (41069042 decodes as 1440x1920), others do not (42444946: 1920x1440). If the decoder already
    # rotated, rotating again would turn the frame (and its K) 90 deg too far
    auto_rotated = rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE) and H0 > W0
    Ws, Hs = (H0, W0) if auto_rotated else (W0, H0)          # sensor-orientation size
    K0 = arkitscenes_K(video, Ws, Hs)
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
        K = None if K0 is None else _rotate_K(K0, rot, Ws, Hs)
        if rot is not None and not auto_rotated:
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


def load_video(path: Path, work_dir: Path, n_frames: int = 32, scale: float = 1.0, method: str = "mapanything",
               n_dense: int = 200):
    """method="sfm" (opt-in, NOT default: see below): COLMAP poses on n_dense frames (SIFT, sequential matching, known K fixed),
    MapAnything dense depth on n_frames of the registered frames, fused (recon.fuse_sfm_depth): SfM
    fixes MapAnything's inconsistent poses, MapAnything supplies metric depth and scale.
    method="mapanything" (default): MapAnything alone on n_frames keyframes.

    Why sfm is not the default (42444946, 2026-10-04): with sensitive SIFT COLMAP registers 93/200
    (incremental) or 157/200 (global) frames, but every camera gets the same projection centre: the
    walk is reconstructed as a pure-rotation panorama (most pairs are classified planar/panoramic:
    low-texture walls seen up close). Checked against ARKit's trajectory (9.5 m path): Sim3 ATE
    41-63 cm. Fused output: ceiling 2.67 vs 3.06 m. Research on the degeneracy is pending."""
    import json
    video = find_video(path)
    work_dir = Path(work_dir)
    if method == "sfm":
        dense, ts_d = extract_keyframes(video, work_dir / "frames_dense", n=n_dense, long_side=1024)
        Ks = json.loads((work_dir / "frames_dense" / "intrinsics.json").read_text()) \
            if (work_dir / "frames_dense" / "intrinsics.json").exists() else {}
        K = np.asarray(Ks[dense[0].name]) if dense and dense[0].name in Ks else None
        sfm = sfm_poses(dense, work_dir / "sfm", K)
        reg = [i for i, p in enumerate(dense) if p.name in sfm]
        print(f"SfM registered {len(reg)}/{len(dense)} frames")
        if len(reg) >= 8:
            pick = [reg[int(round(j))] for j in np.linspace(0, len(reg) - 1, min(n_frames, len(reg)))]
            paths = [dense[i] for i in pick]
            views = run_mapanything(paths, cache=work_dir / "cache")
            sizes = [cv2.imread(str(p)).shape[1::-1] for p in paths]
            Kk = [np.asarray(Ks[p.name]) if p.name in Ks else None for p in paths]
            fused, kept, diag = fuse_sfm_depth(views, [p.name for p in paths], sfm, sizes, Kk)
            print("fusion:", diag)
            if len(fused) >= 8:
                return to_capture(fused, ts_d[pick][kept], Path(path), scale=scale)
        print("SfM too sparse; falling back to MapAnything alone")
    paths, ts = extract_keyframes(video, work_dir / "frames", n=n_frames)
    views = run_mapanything(paths, cache=work_dir / "cache")
    return to_capture(views, ts, Path(path), scale=scale)
