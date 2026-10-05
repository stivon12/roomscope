"""Repeatability without ground truth: two captures of the same space must agree within their intervals.

    python bench/repeatability.py out/yc/1a8384c3f6 out/yc/c7d28f72c6 [--out out/repeatability.json]

1. Register capture B's fused cloud (cloud.npz, result frame) to A's: both frames are gravity-up and
   Manhattan-aligned, so only a yaw in {0, 90, 180, 270} deg and a translation are unknown. Each yaw is
   tried from the centroid offset and refined by point-to-plane ICP; the best fitness wins.
2. Rooms are matched one-to-one by polygon IoU > 0.5 (greedy).
3. For each matched room: floor area, observed ceiling height, and walls whose transformed segments lie
   on the same line (same axis, < 0.15 m apart, > 50 % overlap) are compared.
   Agreement = |a - b| <= half_a + half_b: if each interval covers the truth, the two must overlap. A
   difference larger than both half-widths together means at least one interval is wrong.
Unmatched rooms (the partitions differ) are listed, not hidden.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from shapely.affinity import affine_transform
from shapely.geometry import LineString, Polygon


def _pcd(d: Path) -> o3d.geometry.PointCloud:
    c = np.load(d / "cloud.npz")
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c["P"].astype(np.float64)))
    p.normals = o3d.utility.Vector3dVector(c["N"].astype(np.float64))
    return p


def register(a: Path, b: Path) -> tuple[np.ndarray, float, float]:
    """4x4 transform taking B's result frame into A's, ICP fitness and inlier RMSE."""
    A, B = _pcd(a), _pcd(b)
    ca, cb = np.asarray(A.points).mean(0), np.asarray(B.points).mean(0)
    best = None
    for k in range(4):
        th = k * np.pi / 2
        R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = ca - R @ cb
        for dist in (0.5, 0.2, 0.08):
            r = o3d.pipelines.registration.registration_icp(
                B, A, dist, T, o3d.pipelines.registration.TransformationEstimationPointToPlane())
            T = r.transformation
        if best is None or r.fitness > best[1]:
            best = (T, float(r.fitness), float(r.inlier_rmse))
    return best


def _xf(T: np.ndarray):
    return [T[0, 0], T[0, 1], T[1, 0], T[1, 1], T[0, 3], T[1, 3]]


def _agree(ma: dict, mb: dict) -> dict:
    a, b = ma["value"], mb["value"]
    ha, hb = ma["hi"] - a, mb["hi"] - b
    return {"a": round(a, 4), "b": round(b, 4), "diff": round(b - a, 4), "half_a": round(ha, 4),
            "half_b": round(hb, 4), "agree": bool(abs(b - a) <= ha + hb)}


def compare(a: Path, b: Path) -> dict:
    T, fit, rmse = register(a, b)
    ra, rb = (json.loads((d / "result.json").read_text()) for d in (a, b))
    PA = {r["id"]: Polygon(r["polygon"]) for r in ra["rooms"]}
    PB = {r["id"]: affine_transform(Polygon(r["polygon"]), _xf(T)) for r in rb["rooms"]}
    cand = sorted(((PA[i].intersection(PB[j]).area / PA[i].union(PB[j]).area, i, j) for i in PA for j in PB),
                  reverse=True)
    pairs, ua, ub = [], set(), set()
    for iou, i, j in cand:
        if iou > 0.5 and i not in ua and j not in ub:
            pairs.append((i, j, iou)); ua.add(i); ub.add(j)
    RA = {r["id"]: r for r in ra["rooms"]}
    RB = {r["id"]: r for r in rb["rooms"]}
    rooms, checks = [], []
    for i, j, iou in pairs:
        x, y = RA[i], RB[j]
        row = {"a": i, "b": j, "iou": round(iou, 3), "floor_area": _agree(x["floor_area"], y["floor_area"])}
        checks.append(("floor_area", row["floor_area"]))
        if x["ceiling_height"].get("observed", True) and y["ceiling_height"].get("observed", True):
            row["ceiling_height"] = _agree(x["ceiling_height"], y["ceiling_height"])
            checks.append(("ceiling_height", row["ceiling_height"]))
        walls = []
        for wa in x["walls"]:
            la = LineString([wa["start"], wa["end"]])
            for wb in y["walls"]:
                lb = affine_transform(LineString([wb["start"], wb["end"]]), _xf(T))
                (ax0, ay0), (ax1, ay1) = la.coords
                (bx0, by0), (bx1, by1) = lb.coords
                va, vb = abs(ax1 - ax0) < abs(ay1 - ay0), abs(bx1 - bx0) < abs(by1 - by0)
                if va != vb:
                    continue
                off = abs(ax0 - bx0) if va else abs(ay0 - by0)
                sa = sorted((ay0, ay1) if va else (ax0, ax1))
                sb = sorted((by0, by1) if vb else (bx0, bx1))
                ov = min(sa[1], sb[1]) - max(sa[0], sb[0])
                if off < 0.15 and ov > 0.5 * max(sa[1] - sa[0], sb[1] - sb[0]):
                    w = {**_agree(wa["length"], wb["length"]), "id_a": wa["id"], "id_b": wb["id"]}
                    walls.append(w)
                    checks.append(("wall_length", w))
        row["walls"] = walls
        rooms.append(row)
    summary = {}
    for kind in ("floor_area", "ceiling_height", "wall_length"):
        c = [x for k, x in checks if k == kind]
        if c:
            d = np.abs([x["diff"] for x in c])
            summary[kind] = {"n": len(c), "agree": int(sum(x["agree"] for x in c)),
                             "median_abs_diff": round(float(np.median(d)), 4),
                             "median_half_sum": round(float(np.median([x["half_a"] + x["half_b"] for x in c])), 4)}
    return {"a": str(a), "b": str(b), "icp_fitness": round(fit, 3), "icp_rmse_m": round(rmse, 4),
            "matched_rooms": rooms, "unmatched_a": sorted(set(PA) - ua), "unmatched_b": sorted(set(PB) - ub),
            "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/repeatability.json"))
    args = ap.parse_args()
    r = compare(args.a, args.b)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=1))
    print(f"ICP fitness {r['icp_fitness']}, rmse {100 * r['icp_rmse_m']:.1f} cm; "
          f"matched {len(r['matched_rooms'])} rooms, unmatched A {r['unmatched_a']} B {r['unmatched_b']}")
    for m in r["matched_rooms"]:
        fa = m["floor_area"]
        print(f"  {m['a']}~{m['b']} IoU {m['iou']}: area {fa['a']:.2f} vs {fa['b']:.2f} m2 "
              f"({'agree' if fa['agree'] else 'DISAGREE'}), {len(m['walls'])} walls matched")
    for k, s in r["summary"].items():
        print(f"  {k}: {s['agree']}/{s['n']} agree within their intervals; median |diff| {s['median_abs_diff']}, "
              f"median half-width sum {s['median_half_sum']}")


if __name__ == "__main__":
    main()
