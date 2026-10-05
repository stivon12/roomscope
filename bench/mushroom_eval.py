"""Score a run on a converted MuSHRoom capture (bench/mushroom_to_stray.py) against the room's Faro scan.

    python bench/mushroom_eval.py out/mushroom/vr_room_long benchmark/raw/mushroom/room_datasets/vr_room/gt_pd.ply

gt_pd.ply is the Faro Focus 3D X130 point cloud (with normals) in its own frame. icp_iphone.json is NOT used:
it maps the authors' COLMAP poses (not the Polycam poses we run on) to the Faro frame, and its z scale is
1.0063, so it is not rigid. The scan is registered rigidly into the result frame by the same code as for the
ARKitScenes laser (eval/laser.register_to + ICP), and scored by eval/laser.score_result: ceiling height at
the centroid, wall-plane offsets and wall lengths, each with whether our interval covers it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from roomscope.eval.laser import refine_registration, register_to, score_result, summarise


def load_faro(ply: Path, voxel: float = 0.01) -> o3d.geometry.PointCloud:
    """The file's normal fields are all zero (checked on vr_room), so normals are estimated and oriented
    toward the room's interior (bounding-box centre at mid-height): floor up, ceiling down, walls inward,
    the convention score_result expects. Holds for a single convex-ish room, which each MuSHRoom scene is."""
    pc = o3d.io.read_point_cloud(str(ply)).voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    pc.orient_normals_towards_camera_location(pc.get_axis_aligned_bounding_box().get_center())
    return pc


def score(out_dir: Path, ply: Path) -> dict:
    result = json.loads((out_dir / "result.json").read_text())
    cl = np.load(out_dir / "cloud.npz")
    P, N = cl["P"].astype(np.float64), cl["N"].astype(np.float64)
    ref = load_faro(ply)
    T0, _ = register_to(ref.voxel_down_sample(0.05), P, N)
    T, fit = refine_registration(ref.voxel_down_sample(0.02), P, N, T0)
    ref.transform(T)
    R = np.concatenate([np.asarray(rm["polygon"]) for rm in result["rooms"]])
    lo = np.r_[R.min(0) - 0.5, P[:, 2].min() - 0.3]
    hi = np.r_[R.max(0) + 0.5, P[:, 2].max() + 0.3]
    Lp = np.asarray(ref.points)
    ref = ref.select_by_index(np.where(np.all((Lp > lo) & (Lp < hi), axis=1))[0])
    sc = score_result(result, ref)
    sc.update({"reference": f"faro:{ply}", "icp_fitness": fit, "n_ref_planes": len(sc["wall_planes"]),
               "floor_area": result["rooms"][0]["floor_area"] if len(result["rooms"]) == 1 else None})
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("ply", type=Path)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    s = score(a.out_dir, a.ply)
    print(f"icp_fitness={s['icp_fitness']:.3f} walls_with_ref={s['n_ref_planes']}")
    print(summarise(s))
    if a.v:
        for k in ("ceil", "walls", "wall_planes"):
            for r in s[k]:
                print(k, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()})
    (a.out_dir / "faro_score.json").write_text(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
