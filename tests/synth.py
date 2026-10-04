"""Synthetic Stray Scanner capture of a known multi-room layout.

Why: real captures are not in yet, and even once they are, a capture with *exact* ground truth lets us
separate algorithm error from sensor/GT error. We ray-cast LiDAR-like depth from a walking camera
through a hand-specified floor plan and write the exact Stray Scanner export format:

    depth/000000.npy        uint16 millimetres, 192x256 (H x W)
    confidence/000000.npy   uint8 0/1/2
    odometry.csv            timestamp, frame, x, y, z, qx, qy, qz, qw   (ARKit camera-to-world, y-up, camera looks -Z)
    camera_matrix.csv       3x3 intrinsics of the 1920x1440 RGB stream

Our world frame is z-up (floor z=0). ARKit's is y-up: (x_a, y_a, z_a) = (x, z, -y).

Usage:  python tests/synth.py <out_dir> [--drift] [--seed N]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation

CEIL = 2.5
WALL_T = 0.1
RGB_W, RGB_H = 1920, 1440
DEP_W, DEP_H = 256, 192
FX = FY = 1500.0
CX, CY = RGB_W / 2, RGB_H / 2


@dataclass
class Opening:
    id: str
    type: str          # door | window
    wall: str          # 'S','E','N','W' side of the room
    lo: float          # world coordinate along the wall where the gap starts
    hi: float
    sill: float = 0.0
    top: float = 2.03
    connects_to: str | None = None


@dataclass
class Room:
    id: str
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    openings: list[Opening] = field(default_factory=list)


# Layout (metres). Hallway connector along y in [0, 1.2]; rooms A, B north of it, C south of it.
# Interior walls are 0.1 m thick.
ROOMS = [
    Room("hall", "Hallway", 0.0, 0.0, 6.0, 1.2, [
        Opening("D1", "door", "N", 1.00, 1.85, connects_to="roomA"),
        Opening("D2", "door", "N", 4.20, 5.00, connects_to="roomB"),
        Opening("D3", "door", "S", 1.50, 2.40, connects_to="roomC"),
    ]),
    Room("roomA", "Bedroom", 0.0, 1.3, 3.0, 4.8, [
        Opening("D1", "door", "S", 1.00, 1.85, connects_to="hall"),
        Opening("N1", "window", "W", 2.50, 3.70, sill=0.90, top=2.10),
    ]),
    Room("roomB", "Living", 3.1, 1.3, 6.0, 5.3, [
        Opening("D2", "door", "S", 4.20, 5.00, connects_to="hall"),
        Opening("N2", "window", "N", 4.00, 5.40, sill=0.80, top=2.10),
    ]),
    Room("roomC", "Kitchen", 0.0, -3.6, 4.0, -0.1, [
        Opening("D3", "door", "N", 1.50, 2.40, connects_to="hall"),
        Opening("N3", "window", "E", -2.80, -1.60, sill=1.00, top=2.10),
    ]),
]


def _quad(verts, tris_list, a, b, c, d):
    """Append quad a-b-c-d (two triangles)."""
    base = len(verts)
    verts.extend([a, b, c, d])
    tris_list.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])


def _wall_face_rects(room: Room, side: str):
    """Rectangles (u0,u1,v0,v1) covering a wall face minus its openings. u is the world coordinate along the wall."""
    if side in "SN":
        u0, u1 = room.x0, room.x1
    else:
        u0, u1 = room.y0, room.y1
    gaps = sorted([(o.lo, o.hi, o.sill, o.top) for o in room.openings if o.wall == side])
    rects, cur = [], u0
    for lo, hi, sill, top in gaps:
        rects.append((cur, lo, 0.0, CEIL))
        if sill > 0:
            rects.append((lo, hi, 0.0, sill))
        rects.append((lo, hi, top, CEIL))
        cur = hi
    rects.append((cur, u1, 0.0, CEIL))
    return [r for r in rects if r[1] - r[0] > 1e-6 and r[3] - r[2] > 1e-6]


def build_mesh() -> o3d.t.geometry.TriangleMesh:
    verts, tris = [], []
    for r in ROOMS:
        # floor and ceiling
        _quad(verts, tris, [r.x0, r.y0, 0], [r.x1, r.y0, 0], [r.x1, r.y1, 0], [r.x0, r.y1, 0])
        _quad(verts, tris, [r.x0, r.y0, CEIL], [r.x1, r.y0, CEIL], [r.x1, r.y1, CEIL], [r.x0, r.y1, CEIL])
        for side in "SNEW":
            for u0, u1, v0, v1 in _wall_face_rects(r, side):
                if side == "S":
                    y = r.y0; _quad(verts, tris, [u0, y, v0], [u1, y, v0], [u1, y, v1], [u0, y, v1])
                elif side == "N":
                    y = r.y1; _quad(verts, tris, [u0, y, v0], [u1, y, v0], [u1, y, v1], [u0, y, v1])
                elif side == "W":
                    x = r.x0; _quad(verts, tris, [x, u0, v0], [x, u1, v0], [x, u1, v1], [x, u0, v1])
                else:
                    x = r.x1; _quad(verts, tris, [x, u0, v0], [x, u1, v0], [x, u1, v1], [x, u0, v1])
        # door jambs + head through the wall thickness (only for doors on the room's S or N side, pointing outward)
        for o in r.openings:
            if o.type != "door" or o.wall not in "SN":
                continue
            y = r.y0 if o.wall == "S" else r.y1
            y2 = y - WALL_T if o.wall == "S" else y + WALL_T
            for xe in (o.lo, o.hi):
                _quad(verts, tris, [xe, y, 0], [xe, y2, 0], [xe, y2, o.top], [xe, y, o.top])
            _quad(verts, tris, [o.lo, y, o.top], [o.hi, y, o.top], [o.hi, y2, o.top], [o.lo, y2, o.top])
    # 'garden fence' 2 m outside the building so window rays return something (as real outdoor returns would)
    _quad(verts, tris, [-2.5, -6, 0], [-2.5, 8, 0], [-2.5, 8, 3], [-2.5, -6, 3])
    _quad(verts, tris, [8.5, -6, 0], [8.5, 8, 0], [8.5, 8, 3], [8.5, -6, 3])
    _quad(verts, tris, [-3, 7.5, 0], [9, 7.5, 0], [9, 7.5, 3], [-3, 7.5, 3])
    _quad(verts, tris, [-3, -6, 0], [9, -6, 0], [9, -6, 3], [-3, -6, 3])
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex.positions = o3d.core.Tensor(np.asarray(verts, np.float32))
    mesh.triangle.indices = o3d.core.Tensor(np.asarray(tris, np.int32))
    return mesh


def trajectory(fps: float = 10.0):
    """Waypoints (x, y) through the rooms, a loop that returns to start. Yaw sweeps to face walls, pitch nods."""
    wps = [
        (0.6, 0.6), (1.4, 0.6), (1.4, 2.0), (0.7, 2.2), (0.7, 4.1), (2.3, 4.1), (2.3, 2.0), (1.4, 2.0),
        (1.4, 0.6), (4.6, 0.6), (4.6, 2.0), (3.8, 2.2), (3.8, 4.6), (5.4, 4.6), (5.4, 2.0), (4.6, 2.0),
        (4.6, 0.6), (1.95, 0.6), (1.95, -0.9), (0.7, -1.0), (0.7, -2.9), (3.3, -2.9), (3.3, -1.0), (1.95, -0.9),
        (1.95, 0.6), (0.6, 0.6),
    ]
    speed = 0.35  # m/s, half walking pace as the protocol asks
    pts = []
    for (xa, ya), (xb, yb) in zip(wps[:-1], wps[1:]):
        n = max(2, int(np.hypot(xb - xa, yb - ya) / speed * fps))
        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append((xa + t * (xb - xa), ya + t * (yb - ya)))
    pts.append(wps[-1])
    pts = np.asarray(pts)
    n = len(pts)
    k = np.arange(n)
    yaw = k * (2 * np.pi / 90.0)            # one full turn every 9 s: sees every wall repeatedly
    pitch = np.deg2rad(30) * np.sin(k * 2 * np.pi / 37.0)  # nod up to ceiling and down to floor
    return pts, yaw, pitch


def cam_to_world_zup(x, y, yaw, pitch, h=1.4):
    """Camera-to-world (our z-up frame). Camera axes (ARKit convention): x right, y up, -z forward."""
    fwd = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    up_w = np.array([0, 0, 1.0])
    right = np.cross(fwd, up_w); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = right, up, -fwd
    T[:3, 3] = [x, y, h]
    return T


# z-up world -> ARKit y-up world
ZUP_TO_ARKIT = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])


def drift_transforms(n, rng, yaw_sigma_deg=0.04, trans_sigma=0.004):
    """Accumulated odometry drift as a random walk: per-frame yaw and xy translation increments (z-up)."""
    dyaw = np.cumsum(rng.normal(0, np.deg2rad(yaw_sigma_deg), n))
    dxy = np.cumsum(rng.normal(0, trans_sigma, (n, 2)), axis=0)
    out = []
    for a, (dx, dy) in zip(dyaw, dxy):
        D = np.eye(4)
        D[:3, :3] = Rotation.from_euler("z", a).as_matrix()
        D[:2, 3] = [dx, dy]
        out.append(D)
    return out


def generate(out: Path, drift: bool = False, seed: int = 0, frame_stride: int = 1) -> Path:
    out = Path(out)
    (out / "depth").mkdir(parents=True, exist_ok=True)
    (out / "confidence").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(build_mesh())

    pts, yaw, pitch = trajectory()
    idx = np.arange(0, len(pts), frame_stride)
    drifts = drift_transforms(len(pts), rng) if drift else [np.eye(4)] * len(pts)

    s = DEP_W / RGB_W
    fx, fy, cx, cy = FX * s, FY * s, CX * s, CY * s
    uu, vv = np.meshgrid(np.arange(DEP_W) + 0.5, np.arange(DEP_H) + 0.5)
    dirs_cam = np.stack([(uu - cx) / fx, -(vv - cy) / fy, -np.ones_like(uu)], -1).reshape(-1, 3)

    rows = []
    for fi, k in enumerate(idx):
        T = cam_to_world_zup(pts[k, 0], pts[k, 1], yaw[k], pitch[k])
        d_world = dirs_cam @ T[:3, :3].T
        rays = np.concatenate([np.repeat(T[None, :3, 3], len(d_world), 0), d_world], 1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays))
        t = ans["t_hit"].numpy()  # along unnormalised dir whose camera z = -1, so t == z-depth
        nrm = ans["primitive_normals"].numpy()
        hit = np.isfinite(t)
        z = np.where(hit, t, 0.0)
        # incidence angle -> confidence, as ARKit lowers confidence at grazing angles and long range
        dn = d_world / np.linalg.norm(d_world, axis=1, keepdims=True)
        cosi = np.abs(np.sum(dn * nrm, 1))
        rng_m = z * np.linalg.norm(dirs_cam, axis=1)
        conf = np.full(z.shape, 2, np.uint8)
        conf[(cosi < 0.25) | (rng_m > 4.0)] = 1
        conf[(rng_m > 5.0) | ~hit] = 0
        noise = rng.normal(0, 1, z.shape) * (0.004 + 0.004 * z)
        z = np.where(hit, z + noise, 0.0)
        z[rng_m > 5.0] = 0.0  # LiDAR max range
        np.save(out / "depth" / f"{fi:06d}.npy", (np.clip(z, 0, 65.0) * 1000).astype(np.uint16).reshape(DEP_H, DEP_W))
        np.save(out / "confidence" / f"{fi:06d}.npy", conf.reshape(DEP_H, DEP_W))

        T_rep = drifts[k] @ T                 # what odometry *reports* (drifted)
        Ta = ZUP_TO_ARKIT @ T_rep             # camera axes already in ARKit convention
        q = Rotation.from_matrix(Ta[:3, :3]).as_quat()  # x, y, z, w
        rows.append(f"{k / 10.0:.6f}, {fi}, {Ta[0, 3]:.6f}, {Ta[1, 3]:.6f}, {Ta[2, 3]:.6f}, "
                    f"{q[0]:.8f}, {q[1]:.8f}, {q[2]:.8f}, {q[3]:.8f}")
    (out / "odometry.csv").write_text("timestamp, frame, x, y, z, qx, qy, qz, qw\n" + "\n".join(rows) + "\n")
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
    np.savetxt(out / "camera_matrix.csv", K, delimiter=",", fmt="%.6f")
    write_gt(out / "gt")
    return out


def write_gt(gt_dir: Path):
    """Ground truth in benchmark/gt/TEMPLATE.yaml format, one file per room."""
    gt_dir.mkdir(parents=True, exist_ok=True)
    for r in ROOMS:
        corners = {"S": (r.x0, r.y0), "E": (r.x1, r.y0), "N": (r.x1, r.y1), "W": (r.x0, r.y1)}
        # walls CCW from the south-west corner: S (x0->x1), E (y0->y1), N (x1->x0), W (y1->y0)
        lengths = {"S": r.x1 - r.x0, "E": r.y1 - r.y0, "N": r.x1 - r.x0, "W": r.y1 - r.y0}
        ops = []
        for o in r.openings:
            start = {"S": r.x0, "E": r.y0, "N": r.x1, "W": r.y1}[o.wall]
            offset = (o.lo - start) if o.wall in "SE" else (start - o.hi)
            d = {"id": o.id, "type": o.type, "wall": o.wall, "width_m": round(o.hi - o.lo, 4),
                 "height_m": round(o.top - o.sill, 4), "offset_m": round(offset, 4)}
            if o.type == "window":
                d["sill_m"] = o.sill
            if o.connects_to:
                d["connects_to"] = o.connects_to
            ops.append(d)
        doc = {
            "room_id": r.id, "name": r.name, "measured_by": "synthetic", "instrument": "exact",
            "ceiling_height_m": [CEIL, CEIL, CEIL],
            "walls": [{"id": s, "length_m": round(lengths[s], 4)} for s in "SENW"],
            "openings": ops,
            "floor_area_m2": round((r.x1 - r.x0) * (r.y1 - r.y0), 4),
            "bbox_world": [r.x0, r.y0, r.x1, r.y1],
            "corners_world": {k: list(v) for k, v in corners.items()},
            "damage": [], "hazards": [],
        }
        (gt_dir / f"{r.id}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    a = ap.parse_args()
    p = generate(a.out, drift=a.drift, seed=a.seed, frame_stride=a.stride)
    print(f"wrote {p}")
