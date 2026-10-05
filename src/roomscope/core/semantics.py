"""Semantic labels on LiDAR points: per-frame indoor segmentation (frontends/seg_worker.py, EoMT-L ADE20K)
projected onto the points each depth frame produced.

- Frames: about one per `interval_s` of the capture (the model takes ~1 s per frame on an M1); points of
  other frames are labelled through the fused cloud (nearest labelled point within `radius_m`).
- RGB for depth frame i: Stray, the rgb.mp4 frame whose number is the odometry `frame` at that timestamp;
  ARKitScenes, the .mov frame nearest in time (frontends/video.frame_times). Both store frames in sensor
  orientation, so each is turned upright for the model from gravity in its pose and the labels turned back.
- Pixel of a point: points are depth-pixel rays (frontends/lidar._frame_points), so projecting them with
  the depth intrinsics gives back their pixel; labels are returned at the depth map's size.
Labels guide which surfaces are structure; every measurement still comes from depth.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

UNKNOWN = -1
MODEL_INFO = {"name": "EoMT-L (DINOv2)", "version": "tue-mps/ade20k_semantic_eomt_large_512", "license": "MIT",
              "use": "indoor semantic labels (ADE20K-150) on LiDAR points: which planes are walls vs furniture"}


def upright_k(pose: np.ndarray) -> int:
    """Counter-clockwise quarter turns (np.rot90 k) that bring world-up to the top of the image. Camera frame:
    x right, y up (frontends/lidar), so world up in camera coordinates says which image edge is 'up'."""
    ux, uy = (pose[:3, :3].T @ np.array([0.0, 0.0, 1.0]))[:2]
    return int(np.argmax([uy, ux, -uy, -ux]))      # up at top / right / bottom / left -> k = 0 / 1 / 2 / 3


def pick_frames(ts: np.ndarray, interval_s: float) -> list[int]:
    out, last = [], -np.inf
    for i, t in enumerate(ts):
        if t - last >= interval_s:
            out.append(i)
            last = t
    return out


def _video_frames(cap) -> tuple[Path, np.ndarray]:
    """(video path, video frame index for every depth frame of the capture)."""
    root = Path(cap.root)
    if (root / "odometry.csv").exists():                           # Stray: odometry `frame` = video frame
        lines = (root / "odometry.csv").read_text().strip().splitlines()
        names = [h.strip() for h in lines[0].split(",")]
        it, ifr = names.index("timestamp"), names.index("frame")
        rows = np.array([[float(r.split(",")[it]), int(r.split(",")[ifr])] for r in lines[1:] if r.strip()])
        j = np.searchsorted(rows[:, 0], cap.timestamps - 1e-6)
        return root / "rgb.mp4", rows[np.clip(j, 0, len(rows) - 1), 1].astype(int)
    from ..frontends.video import find_video, frame_times    # ARKitScenes: <id>.mov + one .pincam per frame
    vid = find_video(root)
    v = cv2.VideoCapture(str(vid))
    n, fps = int(v.get(cv2.CAP_PROP_FRAME_COUNT)), v.get(cv2.CAP_PROP_FPS)
    v.release()
    ft = frame_times(vid, n, fps)
    return vid, np.abs(ft[None, :] - cap.timestamps[:, None]).argmin(1)


def label_frames(cap, cache_dir: Path, interval_s: float = 1.0) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{depth frame index: (label map, confidence map)} at the depth map's size, for ~1 frame per interval_s."""
    sel = pick_frames(cap.timestamps, interval_s)
    vid, vframe = _video_frames(cap)
    H, W = cap.depth_hw
    key = hashlib.sha1(f"{vid.resolve()}|{vid.stat().st_size}|{interval_s}|{H}x{W}|"
                       f"{[int(vframe[i]) for i in sel]}".encode()).hexdigest()[:12]
    cp = Path(cache_dir) / f"seg_{key}.npz"
    if not cp.exists():
        with tempfile.TemporaryDirectory() as td:
            v = cv2.VideoCapture(str(vid))
            paths = []
            for i in sel:
                v.set(cv2.CAP_PROP_POS_FRAMES, int(vframe[i]))
                ok, im = v.read()
                p = Path(td) / f"{i:06d}.jpg"
                cv2.imwrite(str(p), im if ok else np.zeros((H, W, 3), np.uint8))
                paths.append(str(p))
            v.release()
            job = {"images": paths, "rot90": [upright_k(cap.poses[i]) for i in sel], "target": [[W, H]] * len(sel)}
            jp, op = Path(td) / "job.json", Path(td) / "out.npz"
            jp.write_text(json.dumps(job))
            root = Path(__file__).resolve().parents[2]
            subprocess.run([sys.executable, "-m", "roomscope.frontends.seg_worker", str(jp), str(op)], check=True,
                           env={**__import__("os").environ, "PYTHONPATH": str(root)})
            z = dict(np.load(op))
        z["frames"] = np.array(sel)
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, **z)
    z = np.load(cp)
    label_frames.names = list(z["names"])
    return {int(i): (z[f"label_{k}"], z[f"conf_{k}"]) for k, i in enumerate(z["frames"])}


def point_labels(cap, i: int, lab: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label and confidence of every point of depth frame i, by projecting it back to its depth pixel."""
    P = cap.pts_cam[i]
    K = cap.K_depth
    z = np.maximum(-P[:, 2], 1e-6)                                   # camera looks along -z
    u = np.floor(K[0, 0] * P[:, 0] / z + K[0, 2]).astype(int)
    v = np.floor(-K[1, 1] * P[:, 1] / z + K[1, 2]).astype(int)
    H, W = lab.shape
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    L = np.full(len(P), UNKNOWN, np.int16)
    C = np.zeros(len(P), np.uint8)
    L[ok], C[ok] = lab[v[ok], u[ok]], conf[v[ok], u[ok]]
    return L, C


def label_cloud(cloud, cap, corrs, frames: dict, radius_m: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """(label, confidence) for every point of the fused cloud: the nearest labelled point within radius_m
    (in the same Manhattan-aligned frame as the cloud), UNKNOWN where none is that close."""
    R = cloud.R[:3, :3]
    Ps, Ls, Cs = [], [], []
    for i, (lab, conf) in frames.items():
        P, _, _ = cap.world(i, None if corrs is None else corrs[i])
        L, C = point_labels(cap, i, lab, conf)
        keep = L != UNKNOWN
        Ps.append(P[keep] @ R.T); Ls.append(L[keep]); Cs.append(C[keep])
    if not Ps or not sum(len(p) for p in Ps):
        return np.full(len(cloud.P), UNKNOWN, np.int16), np.zeros(len(cloud.P), np.uint8)
    P, L, C = np.concatenate(Ps), np.concatenate(Ls), np.concatenate(Cs)
    d, j = cKDTree(P).query(cloud.P, distance_upper_bound=radius_m)
    hit = np.isfinite(d)
    out_l = np.full(len(cloud.P), UNKNOWN, np.int16)
    out_c = np.zeros(len(cloud.P), np.uint8)
    out_l[hit], out_c[hit] = L[j[hit]], C[j[hit]]
    return out_l, out_c


def on_wall_mask(sem: np.ndarray, names: list[str], on_wall_classes: list[str]) -> np.ndarray:
    """Per point: 1 = a class lying on a wall surface, 0 = anything else (furniture), -1 = unlabelled."""
    ids = [i for i, n in enumerate(names) if n.strip() in set(on_wall_classes)]
    out = np.where(np.isin(sem, ids), 1, 0).astype(np.int8)
    out[sem == UNKNOWN] = -1
    return out
