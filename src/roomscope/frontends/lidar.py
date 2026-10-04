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

from dataclasses import dataclass, field
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
    align: np.ndarray = field(default_factory=lambda: np.eye(4))   # raw capture world -> poses' world (gravity)

    def world(self, i: int, corr: np.ndarray | None = None):
        """Frame i points/normals/camera centre in world, optionally after a 4x4 world correction."""
        T = self.poses[i] if corr is None else corr @ self.poses[i]
        R, t = T[:3, :3], T[:3, 3]
        return self.pts_cam[i] @ R.T + t, self.nrm_cam[i] @ R.T, t


def _find_root(path: Path) -> tuple[Path, list[str]]:
    """Stray recording folder. Stray writes one folder per recording; with several, the longest is used
    and the others are named in a warning (the protocol asks for one recording through all rooms)."""
    path = Path(path)
    if (path / "odometry.csv").exists():
        return path, []
    hits = sorted(path.glob("*/odometry.csv"))
    if not hits:
        raise FileNotFoundError(f"no odometry.csv under {path}")
    if len(hits) == 1:
        return hits[0].parent, []
    sizes = {h.parent: sum(1 for _ in h.open()) for h in hits}
    best = max(sizes, key=sizes.get)
    others = ", ".join(sorted(d.name for d in sizes if d != best))
    return best, [f"{len(hits)} Stray recordings found; used the longest ({best.name}); ignored: {others}"]


def _read_odometry(csv_path: Path) -> dict[str, np.ndarray]:
    """Stray odometry.csv. Header has spaces after the commas ("timestamp, frame, x, y, z, qx, ...");
    v1.3+ adds per-frame fx, fy, cx, cy (+ distortion centre); older exports have 9 columns."""
    lines = csv_path.read_text().strip().splitlines()
    names = [h.strip() for h in lines[0].split(",")]
    data = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:] if ln.strip()], ndmin=2)
    return {n: data[:, i] for i, n in enumerate(names)}


def _rgb_size(root: Path) -> tuple[int, int] | None:
    vid = root / "rgb.mp4"
    if not vid.exists():
        return None
    cap = cv2.VideoCapture(str(vid))
    wh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return wh if wh[0] > 0 else None


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


def load_stray(path: Path, pixel_stride: int = 3, min_conf: int = 2, max_depth: float = 5.0,
               depth_affine: tuple[float, float] | None = None, hz: float = 10.0) -> LidarCapture:
    """Stray Scanner export (github.com/strayrobots/scanner, docs/format.md and OdometryEncoder.swift).

    - Pose: Stray writes q_WA * q_AC with q_AC a 180 deg rotation about x, i.e. camera-to-world with
      OpenCV camera axes (x right, y down, z forward) in ARKit's world (y up). Our camera frame is
      x right, y up, -z forward, so the pose is right-multiplied by CV_TO_OURS.
    - Intrinsics: per frame in odometry.csv (fx, fy, cx, cy) for the RGB image; camera_matrix.csv only
      holds the last frame's. Scaled to the depth map by depth size / real rgb.mp4 size.
    - Frames: joined on the `frame` column (a depth PNG can be missing when ARKit gave no depth);
      thinned to `hz` by timestamp, matching the ~10 Hz the core was tuned on (Stray records at up to
      60 Hz)."""
    root, warns = _find_root(path)
    odo = _read_odometry(root / "odometry.csv")
    K_last = np.loadtxt(root / "camera_matrix.csv", delimiter=",") if (root / "camera_matrix.csv").exists() else None
    frames = odo["frame"].astype(int)
    first = next((root / "depth" / f"{f:06d}" for f in frames
                  if (root / "depth" / f"{f:06d}.png").exists() or (root / "depth" / f"{f:06d}.npy").exists()), None)
    if first is None:
        raise FileNotFoundError(f"no depth frames under {root / 'depth'}")
    Hd, Wd = _read_img(first).shape[:2]
    rgb = _rgb_size(root)
    if rgb is None:
        rgb = (1920, 1440)
        warns.append("rgb.mp4 missing or unreadable: assumed 1920x1440 to scale intrinsics to the depth map")
    sx, sy = Wd / rgb[0], Hd / rgb[1]
    per_frame_K = all(k in odo for k in ("fx", "fy", "cx", "cy"))
    if not per_frame_K and K_last is None:
        raise FileNotFoundError(f"no intrinsics in {root} (odometry.csv has no fx..cy and no camera_matrix.csv)")

    pts, nrms, poses, ts = [], [], [], []
    last_t = -np.inf
    Kd = None
    for row_i, f in enumerate(frames):
        t = odo["timestamp"][row_i]
        if t - last_t < 1.0 / hz - 1e-6:
            continue
        try:
            depth = _read_img(root / "depth" / f"{f:06d}").astype(np.float32) / 1000.0
            conf = _read_img(root / "confidence" / f"{f:06d}")
        except FileNotFoundError:
            continue
        last_t = t
        if per_frame_K:
            fx, fy, cx, cy = (odo[k][row_i] for k in ("fx", "fy", "cx", "cy"))
        else:
            fx, fy, cx, cy = K_last[0, 0], K_last[1, 1], K_last[0, 2], K_last[1, 2]
        Kd = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1.0]])
        if depth_affine is not None:
            depth = np.where(depth > 0, (depth - depth_affine[1]) / depth_affine[0], 0).astype(np.float32)
        P, N = _frame_points(depth, conf, Kd, pixel_stride, min_conf, max_depth)
        pts.append(P); nrms.append(N)
        Tcv = np.eye(4)
        Tcv[:3, :3] = Rotation.from_quat([odo["qx"][row_i], odo["qy"][row_i], odo["qz"][row_i],
                                          odo["qw"][row_i]]).as_matrix()
        Tcv[:3, 3] = [odo["x"][row_i], odo["y"][row_i], odo["z"][row_i]]
        poses.append(ARKIT_TO_ZUP @ Tcv @ CV_TO_OURS)
        ts.append(t)
    if not poses:
        raise RuntimeError(f"no usable frames in {root}")
    cap = LidarCapture(pts, nrms, np.asarray(poses), Kd, np.asarray(ts), root)
    cap.load_warnings = warns
    return cap


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


def _gravity_align(poses: np.ndarray, pts: list[np.ndarray], nrms: list[np.ndarray],
                   up0=(0.0, 0.0, 1.0)) -> np.ndarray:
    """4x4 rotation taking the capture's world 'up' to +z, estimated from horizontal surfaces.

    ARKit world frames are gravity-aligned whatever way the phone is held (ARKitScenes' sky_direction
    only describes image orientation, e.g. 'Left' = device held sideways; it does not affect poses), and
    ARKitScenes stores them z-up. So the prior is world +z, refined by the mean of world normals within
    25 deg of it (floor normals point up; ceiling normals, flipped, too). We do NOT use the camera's own
    up axis: on a sideways-held device it is 90 deg off (this was a real bug on scene 41069042)."""
    up = np.asarray(up0, float)
    sample = range(0, len(poses), max(1, len(poses) // 80))
    Nw = np.concatenate([nrms[i] @ poses[i, :3, :3].T for i in sample])
    # Sanity check on the prior: a handheld phone is held upright or sideways, so one image axis (x or y)
    # points along gravity on average (0.73-0.92 on all 9 benchmark captures). Surface-normal counts are
    # NOT used for this: in rooms where walls outnumber floor+ceiling points they pick a wall axis.
    img_axes = np.abs(poses[:, :3, :2].mean(0))            # columns: image x, image y in world
    k = int(np.argmax(img_axes.max(1)))
    if k != int(np.argmax(np.abs(up))) or img_axes.max() < 0.5:
        raise ValueError(f"gravity prior {up0} disagrees with camera image axes (mean |axis| {img_axes.round(2).T})")
    for _ in range(3):
        d = Nw @ up
        m = np.abs(d) > np.cos(np.deg2rad(25))
        acc = (Nw[m] * np.sign(d[m])[:, None]).sum(0)
        up = acc / np.linalg.norm(acc)
    z = np.array([0, 0, 1.0])
    v = np.cross(up, z)
    s_, c = np.linalg.norm(v), float(up @ z)
    G = np.eye(4)
    if s_ < 1e-9:
        if c < 0:
            G[:3, :3] = np.diag([1, -1, -1])
        return G
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    G[:3, :3] = np.eye(3) + vx + vx @ vx * ((1 - c) / s_ ** 2)
    return G


def load_arkitscenes(path: Path, pixel_stride: int = 3, frame_stride: int = 1, min_conf: int = 2,
                     max_depth: float = 5.0, depth_affine: tuple[float, float] | None = None) -> LidarCapture:
    """depth_affine=(a, b): measured = a * true + b (from eval/depth_bias.py); inverted per pixel.
    None (default) uses depth as recorded."""
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
        if depth_affine is not None:
            depth = np.where(depth > 0, (depth - depth_affine[1]) / depth_affine[0], 0).astype(np.float32)
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
    return LidarCapture(pts, nrms, poses, Kd, np.asarray(ts), root, align=G)


def load_any(path: Path, **kw) -> LidarCapture:
    """Dispatch on layout: Stray (odometry.csv) or ARKitScenes (lowres_depth/ + lowres_wide.traj)."""
    path = Path(path)
    if (path / "odometry.csv").exists() or any(path.glob("*/odometry.csv")):
        return load_stray(path, **kw)
    return load_arkitscenes(path, **kw)
