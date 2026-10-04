"""Ground truth from the ARKitScenes Faro laser scans, and scoring of a result against it.

Why this is a fair reference:
- The laser cloud is millimetre-grade and comes from a different sensor, so it shares no noise with the
  iPhone LiDAR.
- Reference planes are extracted with a *different* algorithm from the pipeline's: Open3D RANSAC
  segment_plane on laser points, versus the pipeline's histogram peaks on phone points. A bug in one is
  unlikely to be mirrored in the other.

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


def _ref_wall_offset(L: np.ndarray, LN: np.ndarray, axis: int, inward: np.ndarray, c: float, lo: float,
                     hi: float, floor_z: float, win: float = 0.20) -> tuple[float, int] | None:
    """Laser reference position of one axis-aligned wall: laser points facing the same way (within 15
    deg), within +-win of the reported line, over the wall's middle stretch and 0.3-2.0 m height; the
    1 cm histogram peak refined by a trimmed mean. Deterministic (no RANSAC)."""
    shrink = min(0.1, 0.25 * (hi - lo))
    m = (LN @ inward > np.cos(np.deg2rad(15))) & (np.abs(L[:, axis] - c) < win)
    m &= (L[:, 1 - axis] > lo + shrink) & (L[:, 1 - axis] < hi - shrink)
    m &= (L[:, 2] > floor_z + 0.3) & (L[:, 2] < floor_z + 2.0)
    x = L[m, axis]
    if len(x) < 200:
        return None
    h, e = np.histogram(x, np.arange(c - win, c + win + 0.01, 0.01))
    off = e[np.argmax(h)] + 0.005
    for _ in range(3):
        sel = np.abs(x - off) < 0.02
        off = float(np.mean(x[sel]))
    return off, int((np.abs(x - off) < 0.02).sum())


def score_result(result: dict, laser_reg: o3d.geometry.PointCloud) -> dict:
    """Per-room ceiling height, per-wall position and per-wall length errors against the laser
    (result frame). Wall lengths use the reference positions of the two walls that bound them."""
    from shapely import contains_xy
    from shapely.geometry import Polygon

    L = np.asarray(laser_reg.points)
    LN = np.asarray(laser_reg.normals)
    out = {"ceil": [], "walls": [], "wall_planes": []}
    for room in result["rooms"]:
        poly = Polygon(room["polygon"])
        inner = poly.buffer(-0.2)
        inside = contains_xy(inner, L[:, 0], L[:, 1])
        fz = L[inside & (LN[:, 2] > 0.95), 2]
        cz = L[inside & (LN[:, 2] < -0.95), 2]
        f_lvl = None
        if len(fz) > 200:
            f_lvl = float(np.median(fz[np.abs(fz - np.percentile(fz, 10)) < 0.03]))
        if f_lvl is not None and len(cz) > 200:
            c_lvl = float(np.median(cz[np.abs(cz - np.percentile(cz, 90)) < 0.03]))
            ref_h = c_lvl - f_lvl
            ch = room["ceiling_height"]
            out["ceil"].append({"room": room["id"], "ref": ref_h, "err": ch["value"] - ref_h,
                                "covered": ch["lo"] <= ref_h <= ch["hi"]})
        if f_lvl is None:
            f_lvl = float(np.percentile(L[LN[:, 2] > 0.95, 2], 5))
        n_c = len(room["polygon"])
        ref = []
        for k in range(n_c):
            p, q = np.array(room["polygon"][k]), np.array(room["polygon"][(k + 1) % n_c])
            t = (q - p) / np.linalg.norm(q - p)
            axis = 0 if abs(t[0]) < abs(t[1]) else 1              # vertical edge: x = const
            inward = np.array([-t[1], t[0], 0.0])                 # CCW polygon: left normal points inside
            c = p[axis]
            lo, hi = sorted((p[1 - axis], q[1 - axis]))
            r = _ref_wall_offset(L, LN, axis, inward, c, lo, hi, f_lvl)
            ref.append(None if r is None else r[0])
            if r is not None:
                out["wall_planes"].append({"room": room["id"], "wall": f"{room['id']}-W{k + 1}",
                                           "offset_err": float(c - r[0]), "n_ref": r[1]})
        for k, w in enumerate(room["walls"]):
            a, b = ref[k - 1], ref[(k + 1) % n_c]
            if a is None or b is None:
                continue
            ref_len = abs(b - a)
            L_ = w["length"]
            out["walls"].append({"room": room["id"], "wall": w["id"], "ref": ref_len, "err": L_["value"] - ref_len,
                                 "rel": L_["value"] / ref_len - 1, "covered": L_["lo"] <= ref_len <= L_["hi"]})
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
            f"planes   {stat(sc['wall_planes'], 'offset_err')}")


def score_capture(out_dir: Path, scene_root: Path) -> dict:
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
    sc.update({"reference": source, "icp_fitness": fit, "n_ref_planes": len(sc["wall_planes"]),
               "drift": result["meta"]["drift_correction"]})
    return sc


if __name__ == "__main__":
    import sys

    s = score_capture(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"reference={s['reference']} icp_fitness={s['icp_fitness']:.3f} walls_with_ref={s['n_ref_planes']}")
    print(summarise(s))
    if "-v" in sys.argv:
        for k in ("ceil", "walls", "wall_planes"):
            for r in s[k]:
                print(k, r)
