"""LiDAR tier front-end: Stray Scanner export -> per-frame posed points with normals (z-up world).

Stray export layout: depth/NNNNNN.(npy|png) uint16 mm, confidence/NNNNNN.(npy|png) uint8 0-2,
odometry.csv (ARKit camera-to-world, y-up), camera_matrix.csv (intrinsics of the RGB stream).

Design choices worth defending:
- Only confidence==2 depth is kept: ARKit marks grazing-angle, long-range and specular returns (mirrors,
  glass, wet floors) as low/medium confidence; dropping them is the first line of defence against those.
- Points are kept *per frame in camera coordinates* rather than fused once, because drift correction
  later re-poses whole fragments; fusing early would bake the drift in.
- Normals come from the depth image (cross product of neighbour differences on a smoothed depth map):
  cheap, and good enough to sort points into floor / ceiling / wall-facing-±x/±y classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# ARKit y-up world -> our z-up world: (x, y, z) = (x_a, -z_a, y_a)
ARKIT_TO_ZUP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])


@dataclass
class LidarCapture:
    pts_cam: list[np.ndarray]      # per frame (N_i, 3) float32, camera frame (x right, y up, -z forward)
    nrm_cam: list[np.ndarray]      # per frame (N_i, 3) float32 unit normals, oriented toward the camera
    poses: np.ndarray              # (F, 4, 4) camera-to-world, z-up world, as reported by odometry
    K_depth: np.ndarray            # 3x3 intrinsics at depth resolution
    timestamps: np.ndarray
    root: Path

    def world(self, i: int, corr: np.ndarray | None = None):
        """Frame i points/normals/camera centre in world, optionally after a 4x4 world correction."""
        T = self.poses[i] if corr is None else corr @ self.poses[i]
        R, t = T[:3, :3], T[:3, 3]
        return self.pts_cam[i] @ R.T + t, self.nrm_cam[i] @ R.T, t


def _find_root(path: Path) -> Path:
    path = Path(path)
    if (path / "odometry.csv").exists():
        return path
    hits = sorted(path.glob("*/odometry.csv"))
    if not hits:
        raise FileNotFoundError(f"no odometry.csv under {path}")
    return hits[0].parent


def _read_img(stem: Path) -> np.ndarray:
    for ext in (".npy", ".png"):
        p = stem.with_suffix(ext)
        if p.exists():
            return np.load(p) if ext == ".npy" else cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    raise FileNotFoundError(stem)


def _normals_from_depth(P: np.ndarray, valid: np.ndarray, step: int = 3) -> np.ndarray:
    """Per-pixel normals from a (H, W, 3) camera-frame point image via central differences."""
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, step:-step] = P[:, 2 * step:] - P[:, :-2 * step]
    dy[step:-step] = P[2 * step:] - P[:-2 * step]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(norm, 1e-9)
    # orient toward camera (camera at origin): n . p < 0
    flip = np.sum(n * P, -1) > 0
    n[flip] *= -1
    ok = valid.copy()
    ok[:, :step] = ok[:, -step:] = False
    ok[:step] = ok[-step:] = False
    # neighbour validity: a normal straddling a depth hole is meaningless
    vv = valid.astype(np.uint8)
    nb = cv2.erode(vv, np.ones((2 * step + 1, 2 * step + 1), np.uint8)) > 0
    return n, ok & nb & (norm[..., 0] > 0)


def load_stray(path: Path, pixel_stride: int = 3, frame_stride: int = 1, min_conf: int = 2,
               max_depth: float = 5.0) -> LidarCapture:
    root = _find_root(path)
    odo = np.genfromtxt(root / "odometry.csv", delimiter=",", names=True, skip_header=0)
    K = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
    d0 = _read_img(root / "depth" / "000000")
    H, W = d0.shape
    rgb_w = 2 * K[0, 2]
    s = W / rgb_w
    Kd = K.copy()
    Kd[:2] *= s
    fx, fy, cx, cy = Kd[0, 0], Kd[1, 1], Kd[0, 2], Kd[1, 2]
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    ray = np.stack([(uu - cx) / fx, -(vv - cy) / fy, -np.ones_like(uu)], -1)  # z-depth 1 rays

    pts, nrms, poses, ts = [], [], [], []
    frames = np.atleast_1d(odo["frame"]).astype(int)
    for row_i in range(0, len(frames), frame_stride):
        f = frames[row_i]
        try:
            depth = _read_img(root / "depth" / f"{f:06d}").astype(np.float32) / 1000.0
            conf = _read_img(root / "confidence" / f"{f:06d}")
        except FileNotFoundError:
            continue
        valid = (depth > 0.1) & (depth < max_depth) & (conf >= min_conf)
        # smooth depth inside valid regions only, so normals are stable (noise ~1 cm per pixel)
        vm = valid.astype(np.float32)
        num = cv2.boxFilter(depth * vm, -1, (5, 5), normalize=False)
        den = cv2.boxFilter(vm, -1, (5, 5), normalize=False)
        ds = np.where(den > 0, num / np.maximum(den, 1e-6), 0)
        n, nok = _normals_from_depth(ray * ds[..., None], valid)
        P = ray * depth[..., None]  # raw (unsmoothed) depth for geometry
        keep = (valid & nok)[::pixel_stride, ::pixel_stride]
        pts.append(P[::pixel_stride, ::pixel_stride][keep].astype(np.float32))
        nrms.append(n[::pixel_stride, ::pixel_stride][keep].astype(np.float32))
        r = odo[row_i]
        Ta = np.eye(4)
        Ta[:3, :3] = Rotation.from_quat([r["qx"], r["qy"], r["qz"], r["qw"]]).as_matrix()
        Ta[:3, 3] = [r["x"], r["y"], r["z"]]
        poses.append(ARKIT_TO_ZUP @ Ta)
        ts.append(r["timestamp"])
    if not poses:
        raise RuntimeError(f"no usable frames in {root}")
    return LidarCapture(pts, nrms, np.asarray(poses), Kd, np.asarray(ts), root)
