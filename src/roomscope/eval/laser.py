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
                        hits = list(up.rglob(f"laser_scanner_point_clouds/{visit}"))
                        if hits:
                            return hits[0]
    hits = list(scene_root.parent.rglob("laser_scanner_point_clouds/*"))
    return hits[0] if len(hits) == 1 else None


def load_laser(laser_dir: Path, voxel: float = 0.02) -> o3d.geometry.PointCloud:
    """Merge all Faro scans of a visit into one cloud (registered by their _pose.txt), normals oriented
    toward each scan's own scanner position so wall facing directions are reliable."""
    merged = o3d.geometry.PointCloud()
    for ply in sorted(Path(laser_dir).glob("*.ply")):
        pc = o3d.io.read_point_cloud(str(ply)).voxel_down_sample(voxel)
        pose = ply.with_name(ply.stem + "_pose.txt")
        origin = np.zeros(3)
        if pose.exists():
            M = np.loadtxt(pose)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = M[:3], M[3]
            pc.transform(T)
            origin = T[:3, 3]
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
        pc.orient_normals_towards_camera_location(origin)
        merged += pc
    if len(merged.points) == 0:
        raise FileNotFoundError(f"no laser .ply in {laser_dir}")
    return merged.voxel_down_sample(voxel)


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
    """Rigid 4x4 taking laser coordinates into the result (ours) frame, plus ICP fitness."""
    L = np.asarray(laser.points)
    LN = np.asarray(laser.normals)
    # both are z-up; floor heights first
    zl = np.percentile(L[LN[:, 2] > 0.9, 2], 5)
    zo = np.percentile(P_ours[N_ours[:, 2] > 0.9, 2], 5)
    R0 = rz(-manhattan_yaw(LN))
    best = (-1, None)
    lo = np.minimum(P_ours[:, :2].min(0), -12) - 2
    size = int(28 / res)
    img_o = _wall_image(P_ours, N_ours, lo[0], lo[1], size, size, res)
    for k in range(4):
        T = rz(k * np.pi / 2) @ R0
        Lr = L @ T[:3, :3].T
        LNr = LN @ T[:3, :3].T
        c_l, c_o = Lr[:, :2].mean(0), P_ours[:, :2].mean(0)
        Lr[:, :2] += c_o - c_l
        img_l = _wall_image(Lr, LNr, lo[0], lo[1], size, size, res)
        (dx, dy), score = cv2.phaseCorrelate(img_l, img_o)
        if score > best[0]:
            Tk = T.copy()
            Tk[:2, 3] = c_o - c_l + np.array([dx, dy]) * res
            Tk[2, 3] = zo - zl
            best = (score, Tk)
    T0 = best[1]
    src = o3d.geometry.PointCloud(laser)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(P_ours)
    tgt.normals = o3d.utility.Vector3dVector(N_ours)
    T = T0
    for d in (0.15, 0.05, 0.02):
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, d, T, o3d.pipelines.registration.TransformationEstimationPointToPlane())
        T = reg.transformation
    return T, float(reg.fitness)


@dataclass
class RefPlane:
    normal: np.ndarray
    d: float              # plane: n . x = d (n unit, points toward the scanner side)
    n_pts: int


def extract_planes(pc: o3d.geometry.PointCloud, max_planes: int = 40, dist: float = 0.01,
                   min_pts: int = 1500) -> list[RefPlane]:
    """Iterative RANSAC on the (registered) laser cloud. Independent of the pipeline's plane finder."""
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


def score_result(result: dict, planes: list[RefPlane], laser_reg: o3d.geometry.PointCloud,
                 match_dist: float = 0.10, match_deg: float = 10.0) -> dict:
    """Per-room ceiling height and per-wall errors against laser reference (result frame)."""
    from shapely import contains_xy
    from shapely.geometry import Polygon

    L = np.asarray(laser_reg.points)
    LN = np.asarray(laser_reg.normals)
    cosm = np.cos(np.deg2rad(match_deg))
    out = {"ceil": [], "walls": [], "wall_planes": []}
    for room in result["rooms"]:
        poly = Polygon(room["polygon"])
        inner = poly.buffer(-0.2)
        inside = contains_xy(inner, L[:, 0], L[:, 1])
        # reference ceiling height: robust plane levels of laser floor/ceiling inside the room footprint
        fz = L[inside & (LN[:, 2] > 0.95), 2]
        cz = L[inside & (LN[:, 2] < -0.95), 2]
        if len(fz) > 200 and len(cz) > 200:
            f_lvl = np.median(fz[np.abs(fz - np.percentile(fz, 10)) < 0.03])
            c_lvl = np.median(cz[np.abs(cz - np.percentile(cz, 90)) < 0.03])
            ref_h = float(c_lvl - f_lvl)
            ch = room["ceiling_height"]
            out["ceil"].append({"room": room["id"], "ref": ref_h, "err": ch["value"] - ref_h,
                                "covered": ch["lo"] <= ref_h <= ch["hi"]})
        # walls: match each polygon edge's plane to a laser plane (same facing, offset within match_dist)
        n_c = len(room["polygon"])
        edge_ref = []
        for k in range(n_c):
            p, q = np.array(room["polygon"][k]), np.array(room["polygon"][(k + 1) % n_c])
            t = (q - p) / np.linalg.norm(q - p)
            inward = np.array([-t[1], t[0], 0.0])          # CCW polygon -> left normal points inside
            mid = np.r_[(p + q) / 2, 1.2]
            best, bd = None, match_dist
            for pl in planes:
                if pl.normal @ inward < cosm:
                    continue
                dist = abs(pl.normal @ mid - pl.d)
                if dist < bd:
                    best, bd = pl, dist
            edge_ref.append(best)
            if best is not None:
                out["wall_planes"].append({"room": room["id"], "wall": f"{room['id']}-W{k + 1}",
                                           "offset_err": float(best.normal @ mid - best.d)})
        for k, w in enumerate(room["walls"]):
            a, b = edge_ref[k - 1], edge_ref[(k + 1) % n_c]
            if a is None or b is None:
                continue
            s, e = np.array(w["start"]), np.array(w["end"])
            t = np.r_[(e - s) / np.linalg.norm(e - s), 0]
            # reference length: distance between the two perpendicular reference planes along the wall
            pa = np.r_[s, 1.2]
            ta = (a.d - a.normal @ pa) / (a.normal @ t)
            tb = (b.d - b.normal @ pa) / (b.normal @ t)
            ref_len = float(abs(tb - ta))
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
        ref = load_laser(laser_dir)
        T, fit = register_to(ref, P, N)
        source = f"faro:{laser_dir.name}"
    else:
        ref = load_highres_reference(scene_root)
        T, fit = refine_registration(ref, P, N, cl["T_world_to_result"])
        source = "highres_depth (Faro mesh rendered at ARKit poses)"
    ref.transform(T)
    # keep the part of the venue the capture actually covers (laser scans span the whole visit)
    lo, hi = P.min(0) - 0.5, P.max(0) + 0.5
    Lp = np.asarray(ref.points)
    ref = ref.select_by_index(np.where(np.all((Lp > lo) & (Lp < hi), axis=1))[0])
    planes = extract_planes(ref)
    sc = score_result(result, planes, ref)
    sc.update({"reference": source, "icp_fitness": fit, "n_ref_planes": len(planes),
               "drift": result["meta"]["drift_correction"]})
    return sc


if __name__ == "__main__":
    import sys

    s = score_capture(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"reference={s['reference']} icp_fitness={s['icp_fitness']:.3f} planes={s['n_ref_planes']}")
    print(summarise(s))
    if "-v" in sys.argv:
        for k in ("ceil", "walls", "wall_planes"):
            for r in s[k]:
                print(k, r)
