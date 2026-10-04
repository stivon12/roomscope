"""LiDAR tier front-end: raw iPhone LiDAR captures -> per-frame posed points with normals (z-up world).

Two loaders produce the same internal frame list (`LidarCapture`):
- `load_stray`: Stray Scanner export (our stock-app capture route). depth/NNNNNN.(npy|png) uint16 mm,
  confidence/NNNNNN.(npy|png) uint8 0-2, odometry.csv (ARKit camera-to-world, y-up world, camera looks -Z),
  camera_matrix.csv (intrinsics of the RGB stream; depth is a scaled-down version of the same camera).
- `load_arkitscenes`: Apple ARKitScenes raw scene (real iPhone/iPad LiDAR with Faro laser ground truth).
  lowres_depth/<vid>_<ts>.png uint16 mm 256x192, confidence/<vid>_<ts>.png, lowres_wide_intrinsics/*.pincam
  ("w h fx fy cx cy"), lowres_wide.traj ("ts rx ry rz tx ty tz", axis-angle; WORLD-TO-CAMERA, inverted here,
  as in ARKitScenes' tenFpsDataLoader.TrajStringToMatrix), OpenCV camera axes (x right, y down, z forward).

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


def _frame_points(depth: np.ndarray, conf: np.ndarray, Kd: np.ndarray, pixel_stride: int, min_conf: int,
                  max_depth: float):
    """One depth frame -> (points, normals) in our camera frame (x right, y up, -z forward)."""
    H, W = depth.shape
    fx, fy, cx, cy = Kd[0, 0], Kd[1, 1], Kd[0, 2], Kd[1, 2]
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    ray = np.stack([(uu - cx) / fx, -(vv - cy) / fy, -np.ones_like(uu)], -1)  # z-depth 1 rays
    valid = (depth > 0.1) & (depth < max_depth) & (conf >= min_conf)
    # smooth depth inside valid regions only, so normals are stable (noise ~1 cm per pixel)
    vm = valid.astype(np.float32)
    num = cv2.boxFilter(depth * vm, -1, (5, 5), normalize=False)
    den = cv2.boxFilter(vm, -1, (5, 5), normalize=False)
    ds = np.where(den > 0, num / np.maximum(den, 1e-6), 0)
    n, nok = _normals_from_depth(ray * ds[..., None], valid)
    P = ray * depth[..., None]  # raw (unsmoothed) depth for geometry
    keep = (valid & nok)[::pixel_stride, ::pixel_stride]
    return (P[::pixel_stride, ::pixel_stride][keep].astype(np.float32),
            n[::pixel_stride, ::pixel_stride][keep].astype(np.float32))


def load_stray(path: Path, pixel_stride: int = 3, frame_stride: int = 1, min_conf: int = 2,
               max_depth: float = 5.0) -> LidarCapture:
    root = _find_root(path)
    odo = np.genfromtxt(root / "odometry.csv", delimiter=",", names=True)
    K = np.loadtxt(root / "camera_matrix.csv", delimiter=",")
    d0 = _read_img(root / "depth" / "000000")
    W = d0.shape[1]
    Kd = K.copy()
    Kd[:2] *= W / (2 * K[0, 2])   # RGB width ~ 2*cx; depth is the same camera scaled down

    pts, nrms, poses, ts = [], [], [], []
    frames = np.atleast_1d(odo["frame"]).astype(int)
    for row_i in range(0, len(frames), frame_stride):
        f = frames[row_i]
        try:
            depth = _read_img(root / "depth" / f"{f:06d}").astype(np.float32) / 1000.0
            conf = _read_img(root / "confidence" / f"{f:06d}")
        except FileNotFoundError:
            continue
        P, N = _frame_points(depth, conf, Kd, pixel_stride, min_conf, max_depth)
        pts.append(P); nrms.append(N)
        r = odo[row_i]
        Ta = np.eye(4)
        Ta[:3, :3] = Rotation.from_quat([r["qx"], r["qy"], r["qz"], r["qw"]]).as_matrix()
        Ta[:3, 3] = [r["x"], r["y"], r["z"]]
        poses.append(ARKIT_TO_ZUP @ Ta)
        ts.append(r["timestamp"])
    if not poses:
        raise RuntimeError(f"no usable frames in {root}")
    return LidarCapture(pts, nrms, np.asarray(poses), Kd, np.asarray(ts), root)


# OpenCV camera (x right, y down, z forward) -> our camera convention (x right, y up, -z forward)
CV_TO_OURS = np.diag([1.0, -1.0, -1.0, 1.0])


def _arkitscenes_root(path: Path) -> Path:
    path = Path(path)
    if (path / "lowres_depth").is_dir():
        return path
    hits = sorted(p.parent for p in path.rglob("lowres_depth") if p.is_dir())
    if not hits:
        raise FileNotFoundError(f"no lowres_depth/ under {path}")
    return hits[0]


def _gravity_align(poses: np.ndarray, pts: list[np.ndarray], nrms: list[np.ndarray]) -> np.ndarray:
    """4x4 rotation taking the capture's world 'up' to +z.

    Initial up = mean camera-up direction (people hold the phone roughly upright), then refined with the
    mean of world normals within 25 deg of it (floor normals point up; ceiling normals, flipped, too).
    Needed because ARKitScenes trajectories are not guaranteed z-up; harmless when they already are."""
    up = np.mean(poses[:, :3, 1], axis=0)        # our camera +y is 'up'; mean camera-up in world
    up /= np.linalg.norm(up)
    for _ in range(2):
        acc = np.zeros(3)
        for i in range(0, len(poses), max(1, len(poses) // 60)):
            Nw = nrms[i] @ poses[i, :3, :3].T
            d = Nw @ up
            m = np.abs(d) > np.cos(np.deg2rad(25))
            acc += (Nw[m] * np.sign(d[m])[:, None]).sum(0)
        if np.linalg.norm(acc) > 0:
            up = acc / np.linalg.norm(acc)
    z = np.array([0, 0, 1.0])
    v = np.cross(up, z)
    s, c = np.linalg.norm(v), float(up @ z)
    G = np.eye(4)
    if s < 1e-9:
        if c < 0:
            G[:3, :3] = np.diag([1, -1, -1])
        return G
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    G[:3, :3] = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
    return G


def load_arkitscenes(path: Path, pixel_stride: int = 3, frame_stride: int = 1, min_conf: int = 2,
                     max_depth: float = 5.0) -> LidarCapture:
    root = _arkitscenes_root(path)
    traj = np.loadtxt(root / "lowres_wide.traj")
    traj_ts = traj[:, 0]
    files = sorted((root / "lowres_depth").glob("*.png"))
    pts, nrms, poses, ts = [], [], [], []
    for f in files[::frame_stride]:
        vid, tstr = f.stem.rsplit("_", 1)
        t = float(tstr)
        j = int(np.argmin(np.abs(traj_ts - t)))
        if abs(traj_ts[j] - t) > 0.005:          # same tolerance as ARKitScenes' own loader
            continue
        pin = None
        for cand in (tstr, f"{t - 0.001:.3f}", f"{t + 0.001:.3f}"):
            p = root / "lowres_wide_intrinsics" / f"{vid}_{cand}.pincam"
            if p.exists():
                pin = p
                break
        cpath = root / "confidence" / f.name
        if pin is None or not cpath.exists():
            continue
        w, h, fx, fy, cx, cy = np.loadtxt(pin)
        depth = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        conf = cv2.imread(str(cpath), cv2.IMREAD_UNCHANGED)
        # intrinsics are for lowres_wide (same 256x192 grid as lowres_depth); rescale defensively
        sx, sy = depth.shape[1] / w, depth.shape[0] / h
        Kd = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1.0]])
        P, N = _frame_points(depth, conf, Kd, pixel_stride, min_conf, max_depth)
        Rwc = cv2.Rodrigues(traj[j, 1:4])[0]
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = Rwc, traj[j, 4:7]
        poses.append(np.linalg.inv(E) @ CV_TO_OURS)   # camera(ours) -> world
        pts.append(P); nrms.append(N); ts.append(t)
    if not poses:
        raise RuntimeError(f"no usable frames in {root}")
    poses = np.asarray(poses)
    G = _gravity_align(poses, pts, nrms)
    poses = np.einsum("ij,fjk->fik", G, poses)
    return LidarCapture(pts, nrms, poses, Kd, np.asarray(ts), root)


def load_any(path: Path, **kw) -> LidarCapture:
    """Dispatch on layout: Stray (odometry.csv) or ARKitScenes (lowres_depth/ + lowres_wide.traj)."""
    path = Path(path)
    if (path / "odometry.csv").exists() or any(path.glob("*/odometry.csv")):
        return load_stray(path, **kw)
    return load_arkitscenes(path, **kw)
