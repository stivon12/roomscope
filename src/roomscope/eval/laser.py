"""Ground truth from the ARKitScenes Faro laser scans, and scoring of a result against it.

Why this is a fair reference:
- The laser cloud is millimetre-grade and comes from a different sensor, so it shares no noise with the
  iPhone LiDAR.
- The reference wall for each of our edges is decided from the laser's own structure (_structural_wall:
  the surface that reaches the ceiling, searched +-1 m across our edge), not as the laser surface nearest
  our edge. The earlier nearest-within-20-cm rule scored our edge against whatever it sat on (a counter
  front, a wardrobe) and hid wrong-surface errors (docs/WALL_ERRORS.md).

The laser lives in its own frame, so it is registered rigidly into the result frame:
both clouds z-up -> Manhattan-align -> 4 yaw hypotheses x 2-D phase correlation of wall occupancy ->
point-to-plane ICP. Rigid only: any scale or drift error in the phone capture stays visible in the
comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d

from ..core.layout import manhattan_yaw, rz


def find_laser_dir(scene_root: Path) -> Path | None:
    """laser_scanner_point_clouds/<visit_id>/ for this video, located via metadata.csv."""
    scene_root = Path(scene_root)
    video_id = scene_root.name
    for up in [scene_root, *scene_root.parents][:5]:
        meta = up / "metadata.csv"
        if meta.exists():
            import csv
            with open(meta) as fh:
                for row in csv.DictReader(fh):
                    if row.get("video_id") == video_id:
                        visit = row.get("visit_id")
                        for base in (up, up.parent):
                            d = base / "laser_scanner_point_clouds" / str(visit)
                            if d.is_dir() and any(d.glob("*.ply")):
                                return d
    hits = list(scene_root.parent.rglob("laser_scanner_point_clouds/*"))
    return hits[0] if len(hits) == 1 else None


PLY_TYPES = {"double": "<f8", "float": "<f4", "uchar": "u1", "uint8": "u1", "int": "<i4", "uint": "<u4",
             "short": "<i2", "ushort": "<u2", "char": "i1"}


def _ply_vertices(path: Path) -> np.memmap:
    """Memory-map the vertex block of a binary little-endian PLY (Faro scans are ~2 GB each, so they
    are never read whole). Returns a structured memmap; reading a slice touches only those bytes."""
    with open(path, "rb") as fh:
        n, props, header_len = None, [], 0
        in_vertex = False
        while True:
            line = fh.readline()
            header_len += len(line)
            t = line.decode("ascii", "replace").split()
            if not t:
                continue
            if t[0] == "format" and t[1] != "binary_little_endian":
                raise ValueError(f"{path}: unsupported PLY format {t[1]}")
            if t[0] == "element":
                in_vertex = t[1] == "vertex"
                if in_vertex:
                    n = int(t[2])
            elif t[0] == "property" and in_vertex:
                if t[1] == "list":
                    raise ValueError("list property in vertex element")
                props.append((t[2], PLY_TYPES[t[1]]))
            elif t[0] == "end_header":
                break
    return np.memmap(path, dtype=np.dtype(props), mode="r", offset=header_len, shape=(n,))


def _scanner_origin(ply: Path) -> np.ndarray:
    """Scanner position from *_pose.txt (4x4, row-vector convention: translation in row 3).

    Verified on visit 421337: the .ply points are ALREADY registered in a common laser frame (two scans
    overlap at 2.4 cm median distance as stored; applying the pose moves them ~1 m apart). So the pose
    is used only to orient normals toward the scanner, never to transform points."""
    pose = ply.with_name(ply.stem + "_pose.txt")
    if not pose.exists():
        return np.zeros(3)
    M = np.loadtxt(pose, delimiter=",") if "," in pose.read_text() else np.loadtxt(pose)
    return np.asarray(M[3, :3], float)


def _unique_scans(laser_dir: Path) -> list[Path]:
    """Each scan appears twice per visit folder (same size, exported in two different frames ~1.6 m
    apart); keep one per size so all kept scans share one frame."""
    seen, out = set(), []
    for ply in sorted(Path(laser_dir).glob("*.ply")):
        sz = ply.stat().st_size
        if sz not in seen:
            seen.add(sz)
            out.append(ply)
    return out


def _voxel(P: np.ndarray, voxel: float) -> np.ndarray:
    q = np.floor(P / voxel).astype(np.int64)
    _, idx = np.unique(q, axis=0, return_index=True)
    return P[idx]


def load_laser(laser_dir: Path, voxel: float = 0.01, bbox: tuple[np.ndarray, np.ndarray] | None = None,
               stride: int = 1, chunk: int = 2_000_000) -> o3d.geometry.PointCloud:
    """Merge the Faro scans of a visit in the common laser frame, streaming one scan at a time.

    Per scan: memory-mapped chunks -> crop to bbox (laser frame) -> voxel downsample;
    then normals oriented toward that scan's scanner position (so wall facing directions are right),
    then merge. Peak memory is one chunk plus one downsampled scan, never a full 2 GB scan."""
    merged = o3d.geometry.PointCloud()
    for ply in _unique_scans(laser_dir):
        V = _ply_vertices(ply)
        origin = _scanner_origin(ply)
        parts = []
        for s0 in range(0, len(V), chunk * stride):
            blk = V[s0:s0 + chunk * stride:stride]
            P = np.c_[blk["x"], blk["y"], blk["z"]].astype(np.float64)
            if bbox is not None:
                keep = np.all((P >= bbox[0]) & (P <= bbox[1]), axis=1)
                P = P[keep]
            if len(P):
                parts.append(_voxel(P, voxel))
        del V
        if not parts:
            continue
        P = _voxel(np.concatenate(parts), voxel)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(P)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=max(4 * voxel, 0.04), max_nn=30))
        pc.orient_normals_towards_camera_location(origin)
        merged += pc
    if len(merged.points) == 0:
        raise FileNotFoundError(f"no laser points in {laser_dir} (bbox={bbox})")
    return merged


def laser_reference(laser_dir: Path, P_ours: np.ndarray, N_ours: np.ndarray, cache_dir: Path,
                    margin: float = 1.0) -> tuple[o3d.geometry.PointCloud, np.ndarray, float]:
    """Registered, cropped, 1 cm laser reference in the RESULT frame, cached per visit.

    1) coarse: every 50th point of each scan at 5 cm -> register to the capture (rigid);
    2) fine: fixed box around the scanners (+-6 m, +-2.5 m), streaming crop at 1 cm, cached once per
       visit so every run scores against byte-identical reference points;
    3) refine registration with ICP on the fine cloud."""
    laser_dir = Path(laser_dir)
    cache = Path(cache_dir) / f"{laser_dir.name}.npz"
    # the visit covers several storeys and returns through windows; the scanned rooms are around the
    # scanner positions (all on one floor), so coarse-register only that part
    origins = np.array([_scanner_origin(p) for p in _unique_scans(laser_dir)])
    c0 = origins.mean(0)
    scan_box = (np.r_[c0[:2] - 10, c0[2] - 2.5], np.r_[c0[:2] + 10, c0[2] + 2.5])
    coarse_cache = Path(cache_dir) / f"{laser_dir.name}_coarse.npz"
    if coarse_cache.exists():
        z = np.load(coarse_cache)
        coarse = o3d.geometry.PointCloud()
        coarse.points = o3d.utility.Vector3dVector(z["P"].astype(np.float64))
        coarse.normals = o3d.utility.Vector3dVector(z["N"].astype(np.float64))
    else:
        coarse = load_laser(laser_dir, voxel=0.05, stride=50, bbox=scan_box)
        coarse_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(coarse_cache, P=np.asarray(coarse.points).astype(np.float32),
                            N=np.asarray(coarse.normals).astype(np.float32))
    T0, _ = register_to(coarse, P_ours, N_ours)          # laser -> result
    # The fine reference is cropped to a FIXED box around the scanners, not to this capture: a
    # capture-dependent crop rewrote the cache between runs and RANSAC on a slightly different point
    # set gave different reference planes (wall errors moved 4 -> 8 cm on identical inputs).
    box = (np.r_[c0[:2] - 6, c0[2] - 2.5], np.r_[c0[:2] + 6, c0[2] + 2.5])
    fine = None
    if cache.exists():
        z = np.load(cache)
        if np.allclose(z["box_lo"], box[0]) and np.allclose(z["box_hi"], box[1]):
            fine = o3d.geometry.PointCloud()
            fine.points = o3d.utility.Vector3dVector(z["P"].astype(np.float64))
            fine.normals = o3d.utility.Vector3dVector(z["N"].astype(np.float64))
    if fine is None:
        fine = load_laser(laser_dir, voxel=0.01, bbox=box)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, P=np.asarray(fine.points).astype(np.float32),
                            N=np.asarray(fine.normals).astype(np.float32), box_lo=box[0], box_hi=box[1])
    T, fit = refine_registration(fine.voxel_down_sample(0.02), P_ours, N_ours, T0)
    fine.transform(T)
    return fine, T, fit


def load_highres_reference(scene_root: Path, frame_step: int = 6, pixel_stride: int = 6,
                           voxel: float = 0.01) -> o3d.geometry.PointCloud:
    """Laser reference when Faro .ply scans are not distributed for a video: ARKitScenes `highres_depth`
    is depth rendered from the Faro laser mesh at each ARKit pose. Back-projecting it with the *same*
    poses reproduces the laser surface exactly, in the raw ARKit world frame (no registration needed).
    Normals are oriented toward the capturing camera."""
    import cv2 as _cv2
    root = Path(scene_root)
    if not (root / "highres_depth").is_dir():
        hits = sorted(p.parent for p in root.rglob("highres_depth") if p.is_dir())
        if not hits:
            raise FileNotFoundError(f"no highres_depth/ under {scene_root}")
        root = hits[0]
    traj = np.loadtxt(root / "lowres_wide.traj")
    files = sorted((root / "highres_depth").glob("*.png"))[::frame_step]
    merged = o3d.geometry.PointCloud()
    for f in files:
        vid, tstr = f.stem.rsplit("_", 1)
        t = float(tstr)
        j = int(np.argmin(np.abs(traj[:, 0] - t)))
        if abs(traj[j, 0] - t) > 0.005:
            continue
        pins = [root / "lowres_wide_intrinsics" / f"{vid}_{c}.pincam" for c in (tstr, f"{t - 0.001:.3f}", f"{t + 0.001:.3f}")]
        pin = next((p for p in pins if p.exists()), None)
        if pin is None:
            continue
        w, h, fx, fy, cx, cy = np.loadtxt(pin)
        d = _cv2.imread(str(f), _cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        H, W = d.shape
        sx, sy = W / w, H / h                      # same camera, higher resolution
        v, u = np.mgrid[0:H:pixel_stride, 0:W:pixel_stride]
        z = d[v, u]
        ok = z > 0.1
        x = ((u[ok] + 0.5) - cx * sx) / (fx * sx) * z[ok]
        y = ((v[ok] + 0.5) - cy * sy) / (fy * sy) * z[ok]
        Pc = np.c_[x, y, z[ok]]                    # OpenCV camera frame
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = _cv2.Rodrigues(traj[j, 1:4])[0], traj[j, 4:7]
        Tcw = np.linalg.inv(E)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(Pc @ Tcw[:3, :3].T + Tcw[:3, 3])
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
        pc.orient_normals_towards_camera_location(Tcw[:3, 3])
        merged += pc
        merged = merged.voxel_down_sample(voxel) if len(merged.points) > 3_000_000 else merged
    if len(merged.points) == 0:
        raise RuntimeError(f"no usable highres_depth frames in {root}")
    return merged.voxel_down_sample(voxel)


def refine_registration(ref: o3d.geometry.PointCloud, P_ours: np.ndarray, N_ours: np.ndarray,
                        T0: np.ndarray) -> tuple[np.ndarray, float]:
    """Point-to-plane ICP from a known initial transform (coarse-to-fine correspondence distances)."""
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(P_ours)
    tgt.normals = o3d.utility.Vector3dVector(N_ours)
    T = T0
    for d in (0.15, 0.05, 0.02):
        reg = o3d.pipelines.registration.registration_icp(
            ref, tgt, d, T, o3d.pipelines.registration.TransformationEstimationPointToPlane())
        T = reg.transformation
    return T, float(reg.fitness)


def _wall_image(P: np.ndarray, N: np.ndarray, x0: float, y0: float, w: int, h: int, res: float) -> np.ndarray:
    m = np.abs(N[:, 2]) < 0.3
    img = np.zeros((h, w), np.float32)
    i = ((P[m, 1] - y0) / res).astype(int)
    j = ((P[m, 0] - x0) / res).astype(int)
    ok = (i >= 0) & (i < h) & (j >= 0) & (j < w)
    np.add.at(img, (i[ok], j[ok]), 1.0)
    return np.minimum(img, 5.0)


def register_to(laser: o3d.geometry.PointCloud, P_ours: np.ndarray, N_ours: np.ndarray,
                res: float = 0.05) -> tuple[np.ndarray, float]:
    """Rigid 4x4 taking laser coordinates into the result (ours) frame, plus ICP fitness.

    Both clouds are gravity-aligned (z up). Floor heights give z; Manhattan yaw leaves a 4-fold
    ambiguity, resolved together with the xy shift by phase correlation of top-down wall-occupancy
    images; then point-to-plane ICP. Rigid only: scale or drift errors in the capture stay visible."""
    L = np.asarray(laser.points)
    LN = np.asarray(laser.normals)
    zl = np.percentile(L[LN[:, 2] > 0.9, 2], 5)
    zo = np.percentile(P_ours[N_ours[:, 2] > 0.9, 2], 5)
    R0 = rz(-manhattan_yaw(LN))
    c_o = P_ours[:, :2].mean(0)
    best = (-1.0, None)
    for k in range(4):
        T = rz(k * np.pi / 2) @ R0
        Lr = L @ T[:3, :3].T
        LNr = LN @ T[:3, :3].T
        shift0 = c_o - Lr[:, :2].mean(0)
        Lr[:, :2] += shift0
        lo = np.minimum(Lr[:, :2].min(0), P_ours[:, :2].min(0)) - 2
        hi = np.maximum(Lr[:, :2].max(0), P_ours[:, :2].max(0)) + 2
        size = int(np.max(hi - lo) / res) + 1
        img_l = _wall_image(Lr, LNr, lo[0], lo[1], size, size, res)
        img_o = _wall_image(P_ours, N_ours, lo[0], lo[1], size, size, res)
        (dx, dy), score = cv2.phaseCorrelate(img_l, img_o)
        if score > best[0]:
            Tk = T.copy()
            Tk[:2, 3] = shift0 + np.array([dx, dy]) * res
            Tk[2, 3] = zo - zl
            best = (score, Tk)
    T0 = best[1]
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(P_ours)
    tgt.normals = o3d.utility.Vector3dVector(N_ours)
    T = T0
    reg = None
    for d in (0.3, 0.1, 0.04):
        reg = o3d.pipelines.registration.registration_icp(
            laser, tgt, d, T, o3d.pipelines.registration.TransformationEstimationPointToPlane())
        T = reg.transformation
    return T, float(reg.fitness)


@dataclass
class RefPlane:
    normal: np.ndarray
    d: float              # plane: n . x = d (n unit, points toward the scanner side)
    n_pts: int


def extract_planes(pc: o3d.geometry.PointCloud, max_planes: int = 80, dist: float = 0.01,
                   min_pts: int = 1500, seed: int = 0) -> list[RefPlane]:
    """Iterative RANSAC on the (registered) laser cloud. Independent of the pipeline's plane finder.
    Seeded: unseeded RANSAC changed which walls got a reference between identical runs."""
    o3d.utility.random.seed(seed)
    rest = o3d.geometry.PointCloud(pc)
    planes = []
    for _ in range(max_planes):
        if len(rest.points) < min_pts:
            break
        (a, b, c, d), idx = rest.segment_plane(dist, 3, 2000)
        if len(idx) < min_pts:
            break
        n = np.array([a, b, c])
        sub = rest.select_by_index(idx)
        mean_n = np.asarray(sub.normals).mean(0)
        if n @ mean_n < 0:      # orient as the laser normals do (toward the scanned side)
            n, d = -n, -d
        planes.append(RefPlane(n, -d, len(idx)))
        rest = rest.select_by_index(idx, invert=True)
    return planes


def _structural_wall(L: np.ndarray, LN: np.ndarray, axis: int, inward: np.ndarray, c: float, lo: float,
                     hi: float, floor_z: float, ceil_z: float, depth: float = 1.0, bin_m: float = 0.10,
                     min_h: float = 0.4, top_m: float = 0.3) -> dict | None:
    """Laser reference for one axis-aligned edge of our polygon, decided from the laser's own structure.

    Our edge only says which wall is meant (its along-wall span and facing direction); the laser surface is
    searched over +-`depth` m across it, so where our edge sits does not choose the answer. Every laser surface
    facing the room (normals within 15 deg) is a candidate; per 10 cm stretch of the wall it is *tall* if it
    covers >= `min_h` of height there and *reaches the ceiling* if it also comes within `top_m` of the ceiling
    above it (downward-facing returns over the stretch). Structure is what reaches the ceiling:
    counters, blinds, sills, sofas and anything seen through a window do not.
    - A ceiling-reaching surface in front of another is an occluder (wardrobe, curtain) when the one behind
      continues on both sides of it.
    - Otherwise each remaining ceiling-reaching surface is a level of the wall, and so is a tall surface behind
      the chosen level over stretches where the chosen level is not seen (a wall whose top is hidden).
    Where the laser did not scan the tops of the walls (MuSHRoom vr_room: walls to 2.45 m under a 3.5 m
    ceiling) nothing qualifies and the edge is "partial": unscored, not guessed (a floor-to-door-height rule
    was tried and accepts a closed door in its reveal as the wall).
    status: "wall" = one level (scored); "step" = several levels, a step in the wall or an occluder that
    cannot be told apart (reported with every level, not scored); "partial" = no structural surface
    (not scored). None = no laser surface at all."""
    s = float(np.sign(inward[axis]))
    shrink = min(0.1, 0.25 * (hi - lo))
    nb = max(1, int(np.ceil((hi - lo - 2 * shrink) / bin_m)))
    d_all = s * (c - L[:, axis])                               # > 0: behind our edge, outside the room
    a_all = L[:, 1 - axis]
    span = (np.abs(d_all) < depth) & (a_all > lo + shrink) & (a_all < hi - shrink)
    b_all = np.clip(((a_all - lo - shrink) / bin_m).astype(int), 0, nb - 1)
    top = np.full(nb, ceil_z)                                   # ceiling over each stretch, above cabinet undersides
    cm = span & (LN[:, 2] < -0.9) & (L[:, 2] > floor_z + 1.8)
    for b in np.unique(b_all[cm]):
        zz = L[cm & (b_all == b), 2]
        if len(zz) >= 10:
            top[b] = np.percentile(zz, 90)
    m = span & (LN @ inward > np.cos(np.deg2rad(15))) & (L[:, 2] > floor_z + 0.3) & (L[:, 2] < ceil_z + 0.3)
    d, z, bi = d_all[m], L[m, 2], b_all[m]
    if len(d) < 50:
        return None
    zc = np.floor((z - floor_z) / 0.05).astype(int)
    edges = np.arange(-depth, depth + 0.01, 0.01)
    h = gaussian_filter1d(np.histogram(d, edges)[0].astype(float), 1.0)
    peaks = sorted((i for i in range(len(h)) if h[i] >= 50 and h[i] == h[max(0, i - 3):i + 4].max()),
                   key=lambda i: -h[i])
    cands = []
    for i in peaks:
        cen = edges[i] + 0.005
        for _ in range(3):
            cen = float(np.mean(d[np.abs(d - cen) < 0.015]))
        if any(abs(cen - k["behind"]) < 0.03 for k in cands):
            continue
        on = np.abs(d - cen) < 0.015
        cells, ztop = np.zeros(nb), np.full(nb, -np.inf)
        for b in np.unique(bi[on]):
            sel = on & (bi == b)
            cells[b] = len(np.unique(zc[sel]))
            ztop[b] = np.percentile(z[sel], 99)
        tall = cells * 0.05 >= min_h
        cands.append({"behind": cen, "tall": tall, "ceil": tall & (ztop > top - top_m), "n": int(on.sum())})
    tall_c = [k for k in cands if k["tall"].sum() >= 3]
    observed = float(np.mean(np.any([k["tall"] for k in cands], 0)))
    struct = [k for k in tall_c if k["ceil"].sum() >= 2]
    if not struct:
        return {"status": "partial", "observed": observed}

    def occluder(f):
        fb = np.where(f["tall"])[0]
        return any((np.where(r["tall"])[0] < fb.min()).any() and (np.where(r["tall"])[0] > fb.max()).any()
                   for r in struct if r["behind"] > f["behind"] + 0.03)
    levels = [k for k in struct if not occluder(k)]
    best = max(levels, key=lambda k: k["tall"].sum())
    levels += [k for k in tall_c if k not in levels and k["behind"] > best["behind"] + 0.03
               and (k["tall"] & ~best["tall"]).sum() >= max(3, 0.5 * k["tall"].sum())]
    others = [k for k in levels if k is not best]
    return {"status": "step" if others else "wall", "offset": c - s * best["behind"],
            "levels": [c - s * k["behind"] for k in others], "n_ref": best["n"],
            "support_m": round(float(best["tall"].sum()) * bin_m, 2), "observed": observed}


def _level_plane(X: np.ndarray, start_pct: float, win: float = 0.04, keep_m: float = 0.02) -> np.ndarray | None:
    """Plane z = a x + b y + c through one horizontal level of laser points: start from the points within
    `win` of the start_pct percentile height (10: the floor under clutter, 90: the main ceiling above
    soffits and fittings), then refit 3 times on points within keep_m of the plane, which follows a tilt
    instead of cutting it."""
    if len(X) < 200:
        return None
    z = X[:, 2]
    keep = np.abs(z - np.percentile(z, start_pct)) < win
    A = np.c_[X[:, :2], np.ones(len(X))]
    for _ in range(3):
        if keep.sum() < 100:
            return None
        co = np.linalg.lstsq(A[keep], z[keep], rcond=None)[0]
        keep = np.abs(z - A @ co) < keep_m
    return co


def _lower_ceiling(C: np.ndarray, cpl: np.ndarray, floor_z: float, cell: float = 0.25, drop: float = 0.15,
                  min_share: float = 0.25) -> dict | None:
    """A second, lower ceiling level: 25 cm cells whose downward-facing laser returns sit >= `drop` below the main
    ceiling plane, covering >= `min_share` of the ceiling cells. Lamps, beams and ducts cover far less."""
    if len(C) < 200:
        return None
    ij = np.floor(C[:, :2] / cell).astype(int)
    _, inv = np.unique(ij, axis=0, return_inverse=True)
    inv = inv.ravel()
    zc = np.array([np.median(C[inv == k, 2]) for k in range(inv.max() + 1)])
    xy = np.array([C[inv == k, :2].mean(0) for k in range(inv.max() + 1)])
    below = (np.c_[xy, np.ones(len(xy))] @ cpl) - zc
    lowc = below >= drop
    if lowc.mean() < min_share:
        return None
    return {"height": float(np.median(zc[lowc]) - floor_z), "share": round(float(lowc.mean()), 2)}


def score_result(result: dict, laser_reg: o3d.geometry.PointCloud) -> dict:
    """Per-room ceiling height, per-wall position and per-wall length errors against the laser
    (result frame). Wall lengths use the reference positions of the two walls that bound them.

    Walls with one structural level are scored (`wall_planes`, `walls`). Walls whose laser reference has a step
    or a hidden rear level go to `wall_planes_step` / `walls_step` with the error to each level and the smallest
    one as a lower bound; walls with no structural surface are listed in `unscored`. Nothing is dropped
    silently: every polygon edge is in exactly one of the three lists."""
    from shapely import contains_xy
    from shapely.geometry import Polygon

    L = np.asarray(laser_reg.points)
    LN = np.asarray(laser_reg.normals)
    out = {"ceil": [], "ceil_step": [], "walls": [], "wall_planes": [], "wall_planes_step": [], "walls_step": [],
           "unscored": []}
    for room in result["rooms"]:
        poly = Polygon(room["polygon"])
        inner = poly.buffer(-0.2)
        inside = contains_xy(inner, L[:, 0], L[:, 1])
        fz = L[inside & (LN[:, 2] > 0.95), 2]
        cz = L[inside & (LN[:, 2] < -0.95), 2]
        f_lvl = None
        if len(fz) > 200:
            f_lvl = float(np.median(fz[np.abs(fz - np.percentile(fz, 10)) < 0.03]))
        # Reference height = ceiling plane minus floor plane at the room centroid, which is what the pipeline
        # reports (core/layout.ceiling_height). Subtracting a high ceiling percentile from a low floor
        # percentile is only right when both are level in the result frame; a 0.6 deg residual tilt
        # (42898811: floor and ceiling both -10 mm/m) inflated the reference by 1.2 cm over a 6 m room.
        fpl = _level_plane(L[inside & (LN[:, 2] > 0.95)], 10)
        cpl = _level_plane(L[inside & (LN[:, 2] < -0.95)], 90)
        if fpl is not None and cpl is not None:
            c = np.array([poly.centroid.x, poly.centroid.y, 1.0])
            ref_h = float(cpl @ c - fpl @ c)
            ch = room["ceiling_height"]
            low = _lower_ceiling(L[inside & (LN[:, 2] < -0.95)], cpl, fpl @ c)
            if low is not None:
                # the laser ceiling has a second level over a large part of the room (soffit, lowered section):
                # one reference height is ambiguous, as for a stepped wall; reported with both levels, not scored
                errs = [ch["value"] - ref_h, ch["value"] - low["height"]]
                out["ceil_step"].append({"room": room["id"], "levels": [ref_h, low["height"]],
                                         "lower_share": low["share"], "level_errs": errs,
                                         "err_lower_bound": float(min(errs, key=abs))})
        if fpl is not None and cpl is not None and not (out["ceil_step"] and out["ceil_step"][-1]["room"] == room["id"]):
            out["ceil"].append({"room": room["id"], "ref": ref_h, "err": ch["value"] - ref_h,
                                "covered": ch["lo"] <= ref_h <= ch["hi"],
                                "ref_percentile": float(np.median(cz[np.abs(cz - np.percentile(cz, 90)) < 0.03]) - f_lvl)
                                if f_lvl is not None and len(cz) > 200 else None,
                                "laser_tilt_mm_per_m": [round(1000 * float(np.hypot(*fpl[:2])), 1),
                                                        round(1000 * float(np.hypot(*cpl[:2])), 1)]})
        if f_lvl is None:
            f_lvl = float(np.percentile(L[LN[:, 2] > 0.95, 2], 5))
        cen = np.array([poly.centroid.x, poly.centroid.y, 1.0])
        floor_z = float(fpl @ cen) if fpl is not None else f_lvl
        ceil_z = float(cpl @ cen) if cpl is not None else float(np.percentile(L[LN[:, 2] < -0.95, 2], 95))
        n_c = len(room["polygon"])
        ref = []
        for k in range(n_c):
            p, q = np.array(room["polygon"][k]), np.array(room["polygon"][(k + 1) % n_c])
            t = (q - p) / np.linalg.norm(q - p)
            axis = 0 if abs(t[0]) < abs(t[1]) else 1              # vertical edge: x = const
            inward = np.array([-t[1], t[0], 0.0])                 # CCW polygon: left normal points inside
            c = p[axis]
            lo, hi = sorted((p[1 - axis], q[1 - axis]))
            wid = f"{room['id']}-W{k + 1}"
            r = _structural_wall(L, LN, axis, inward, c, lo, hi, floor_z, ceil_z)
            ref.append(r if r is not None and "offset" in r else None)
            if r is None or "offset" not in r:
                out["unscored"].append({"room": room["id"], "wall": wid, "length": round(hi - lo, 2),
                                        "reason": "no laser surface" if r is None else "no structural surface"})
                continue
            row = {"room": room["id"], "wall": wid, "offset_err": float(c - r["offset"]), "n_ref": r["n_ref"],
                   "support_m": r["support_m"]}
            if r["status"] == "wall":
                out["wall_planes"].append(row)
            else:
                errs = [c - o for o in [r["offset"], *r["levels"]]]
                row.update({"level_errs": [round(float(e), 4) for e in errs],
                            "err_lower_bound": float(min(errs, key=abs))})
                out["wall_planes_step"].append(row)
        for k, w in enumerate(room["walls"]):
            a, b = ref[k - 1], ref[(k + 1) % n_c]
            if a is None or b is None:
                continue
            ref_len = abs(b["offset"] - a["offset"])
            if ref_len < 0.1:            # both neighbours took the same laser surface: no length to compare
                out["unscored"].append({"room": room["id"], "wall": w["id"], "length": round(w["length"]["value"], 2),
                                        "reason": "length: both neighbouring references are the same surface"})
                continue
            L_ = w["length"]
            row = {"room": room["id"], "wall": w["id"], "ref": ref_len, "err": L_["value"] - ref_len,
                   "rel": L_["value"] / ref_len - 1, "covered": L_["lo"] <= ref_len <= L_["hi"]}
            if a["status"] == b["status"] == "wall":
                out["walls"].append(row)
            else:
                lens = [abs(y - x) for x in [a["offset"], *a["levels"]] for y in [b["offset"], *b["levels"]]]
                row["err_lower_bound"] = float(min((L_["value"] - x for x in lens), key=abs))
                out["walls_step"].append(row)
    return out


def summarise(sc: dict) -> str:
    def stat(rows, key="err"):
        if not rows:
            return "n=0"
        e = np.array([r[key] for r in rows])
        cov = np.mean([r["covered"] for r in rows]) if "covered" in rows[0] else float("nan")
        return f"n={len(e)} MAE={np.mean(np.abs(e)) * 100:.2f}cm max={np.max(np.abs(e)) * 100:.2f}cm bias={np.mean(e) * 100:+.2f}cm coverage={cov:.2f}"
    return (f"ceiling  {stat(sc['ceil'])}\n"
            f"walls    {stat(sc['walls'])}\n"
            f"planes   {stat(sc['wall_planes'], 'offset_err')}\n"
            f"step     planes {stat(sc['wall_planes_step'], 'err_lower_bound')} (lower bound); "
            f"lengths {stat(sc['walls_step'], 'err_lower_bound')}\n"
            f"unscored {len(sc['unscored'])} edges")


def score_capture(out_dir: Path, scene_root: Path, return_ref: bool = False):
    """Score pipeline output in out_dir (result.json + cloud.npz) against the scene's laser reference."""
    import json

    out_dir, scene_root = Path(out_dir), Path(scene_root)
    result = json.loads((out_dir / "result.json").read_text())
    cl = np.load(out_dir / "cloud.npz")
    P, N = cl["P"].astype(np.float64), cl["N"].astype(np.float64)
    laser_dir = find_laser_dir(scene_root)
    if laser_dir is not None:
        cache_dir = out_dir.parent.parent / "gt_cache" if (out_dir.parent.parent / "gt_cache").exists() \
            else Path(__file__).resolve().parents[3] / "out" / "gt_cache"
        ref, T, fit = laser_reference(laser_dir, P, N, cache_dir)
        source = f"faro:{laser_dir.name}"
    else:
        ref = load_highres_reference(scene_root)
        T, fit = refine_registration(ref, P, N, cl["T_world_to_result"])
        ref.transform(T)
        source = "highres_depth (Faro mesh rendered at ARKit poses)"
    # keep the rooms we report on (+0.5 m), not everything glimpsed through doorways: the reference
    # planes should be the walls being scored
    R = np.concatenate([np.asarray(rm["polygon"]) for rm in result["rooms"]])
    lo = np.r_[R.min(0) - 0.5, P[:, 2].min() - 0.3]
    hi = np.r_[R.max(0) + 0.5, P[:, 2].max() + 0.3]
    Lp = np.asarray(ref.points)
    ref = ref.select_by_index(np.where(np.all((Lp > lo) & (Lp < hi), axis=1))[0])
    sc = score_result(result, ref)
    sc.update({"reference": source, "icp_fitness": fit, "n_ref_planes": len(sc["wall_planes"]), "T_laser_to_result": T.tolist(),
               "drift": result["meta"]["drift_correction"]})
    return (sc, ref) if return_ref else sc


if __name__ == "__main__":
    import sys

    s = score_capture(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"reference={s['reference']} icp_fitness={s['icp_fitness']:.3f} walls_with_ref={s['n_ref_planes']}")
    print(summarise(s))
    if "-v" in sys.argv:
        for k in ("ceil", "walls", "wall_planes", "walls_step", "wall_planes_step", "unscored"):
            for r in s[k]:
                print(k, r)
