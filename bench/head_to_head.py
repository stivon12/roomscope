"""Head-to-head at the LiDAR tier: our plan vs Polycam's own reconstruction of the same room, both vs the Faro laser.

    python bench/head_to_head.py [--rooms honka coffee_room] [--tie 0.005]

Data: MuSHRoom (CC BY 4.0) recorded its iPhone captures WITH Polycam and ships Polycam's export of each long capture
(`iphone/long_capture/polycam_mesh/textured.obj`, header "Created by Polycam", 2023; app version not recorded).
Our pipeline runs on the same capture's frames and depth (bench/mushroom_to_stray.py), so both reconstructions come
from identical sensor data.

Dimensions are those of our plan:
- each wall position;
- each wall length (between its two neighbouring walls);
- the ceiling height.

The truth comes from the laser. Polycam's value for the same dimension is measured on its mesh with the SAME tool
(eval/laser.score_result: the structural wall surface and the ceiling/floor planes), after the same rigid
registration to our result frame. Polycam's free tier exports meshes, not dimensions, so its mesh is measured,
not its measuring tool. Only dimensions with a clean reference in both the laser and Polycam's mesh are shared.

A dimension counts as beat or tie when |our error| <= |Polycam error| + tie (0.5 cm, fixed before running).
Writes bench/results/head_to_head.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

from mushroom_eval import load_faro                                     # noqa: E402
from roomscope.eval.laser import refine_registration, register_to, score_result   # noqa: E402

DATA = ROOT / "benchmark/raw/mushroom/room_datasets"


def load_polycam(room: str, spacing: float = 0.01) -> o3d.geometry.PointCloud:
    """Polycam's mesh sampled to ~1 cm points with triangle normals, oriented toward the room centre (the same
    convention as the laser cloud in mushroom_eval.load_faro)."""
    mesh = o3d.io.read_triangle_mesh(str(DATA / room / "iphone/long_capture/polycam_mesh/textured.obj"))
    # Polycam exports y-up (glTF/OBJ convention); our frames and the registration are z-up: (x, y, z) -> (x, -z, y)
    mesh.rotate(np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]), center=(0, 0, 0))
    mesh.compute_triangle_normals()
    n = int(mesh.get_surface_area() / spacing ** 2)
    pc = mesh.sample_points_uniformly(number_of_points=n, use_triangle_normal=True)
    pc.orient_normals_towards_camera_location(pc.get_axis_aligned_bounding_box().get_center())
    return pc


def registered(pc: o3d.geometry.PointCloud, P: np.ndarray, N: np.ndarray) -> tuple[o3d.geometry.PointCloud, float]:
    T0, _ = register_to(pc.voxel_down_sample(0.05), P, N)
    T, fit = refine_registration(pc.voxel_down_sample(0.02), P, N, T0)
    pc = o3d.geometry.PointCloud(pc)
    pc.transform(T)
    return pc, fit


def crop(pc, result, P):
    R = np.concatenate([np.asarray(rm["polygon"]) for rm in result["rooms"]])
    lo, hi = np.r_[R.min(0) - 0.5, P[:, 2].min() - 0.3], np.r_[R.max(0) + 0.5, P[:, 2].max() + 0.3]
    X = np.asarray(pc.points)
    return pc.select_by_index(np.where(np.all((X > lo) & (X < hi), axis=1))[0])


def compare(room: str, res_dir: Path, tie: float) -> dict:
    result = json.loads((res_dir / "result.json").read_text())
    cl = np.load(res_dir / "cloud.npz")
    P, N = cl["P"].astype(float), cl["N"].astype(float)
    laser, fit_l = registered(load_faro(DATA / room / "gt_pd.ply"), P, N)
    poly, fit_p = registered(load_polycam(room), P, N)
    sl, sp = score_result(result, crop(laser, result, P)), score_result(result, crop(poly, result, P))
    rows = []
    # wall positions: err = ours - ref, so Polycam's position = ours - err_p and its error vs laser = err_l - err_p
    pl = {r["wall"]: r for r in sl["wall_planes"]}
    pp = {r["wall"]: r for r in sp["wall_planes"]}
    for w in sorted(set(pl) & set(pp)):
        ours, theirs = pl[w]["offset_err"], pl[w]["offset_err"] - pp[w]["offset_err"]
        rows.append({"dimension": f"{w} position", "ours_err": ours, "polycam_err": theirs})
    wl = {r["wall"]: r for r in sl["walls"]}
    wp = {r["wall"]: r for r in sp["walls"]}
    for w in sorted(set(wl) & set(wp)):
        rows.append({"dimension": f"{w} length", "ours_err": wl[w]["err"], "polycam_err": wp[w]["ref"] - wl[w]["ref"],
                     "truth_m": wl[w]["ref"]})
    for cl_, cp_ in zip(sl["ceil"], sp["ceil"]):
        rows.append({"dimension": f"{cl_['room']} ceiling height", "ours_err": cl_["err"],
                     "polycam_err": cp_["ref"] - cl_["ref"], "truth_m": cl_["ref"]})
    for r in rows:
        r["beat_or_tie"] = bool(abs(r["ours_err"]) <= abs(r["polycam_err"]) + tie)
    return {"room": room, "capture": res_dir.name, "icp_fitness_laser": fit_l, "icp_fitness_polycam": fit_p,
            "rows": rows, "unshared_walls_laser": len(sl["wall_planes_step"]) + len(sl["unscored"]),
            "unshared_walls_polycam": len(sp["wall_planes_step"]) + len(sp["unscored"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", nargs="+", default=["honka", "coffee_room"])
    ap.add_argument("--tie", type=float, default=0.005)
    ap.add_argument("--out", type=Path, default=ROOT / "out/bench/lidar")
    a = ap.parse_args()
    reports = []
    for room in a.rooms:
        res_dir = next((a.out / f"{room}_long").glob("*/result.json")).parent
        reports.append(compare(room, res_dir, a.tie))
    rows = [r for rep in reports for r in rep["rows"]]
    k = sum(r["beat_or_tie"] for r in rows)
    lines = ["# Head-to-head: roomscope (LiDAR tier) vs Polycam, both vs Faro laser", "",
             "Generated by `python bench/head_to_head.py`. Polycam = the app's own mesh export shipped with MuSHRoom "
             "(same capture as our input; app version not recorded in the dataset). Polycam's mesh is measured with the "
             "same tool as the laser. Beat or tie: |ours| <= |Polycam| + 0.5 cm.", "",
             f"**Beat or tie on {k} of {len(rows)} shared dimensions ({100 * k / max(len(rows), 1):.0f} %; gate: >= 70 %).**", "",
             "| room | dimension | truth (laser) | our error | Polycam error | beat or tie |", "|---|---|---|---|---|---|"]
    for rep in reports:
        for r in rep["rows"]:
            t = f"{r['truth_m']:.3f} m" if "truth_m" in r else "-"
            lines.append(f"| {rep['room']} | {r['dimension']} | {t} | {100 * r['ours_err']:+.1f} cm | "
                         f"{100 * r['polycam_err']:+.1f} cm | {'yes' if r['beat_or_tie'] else 'no'} |")
    lines += ["", "Registration ICP fitness (laser / Polycam): " + "; ".join(
        f"{rep['room']} {rep['icp_fitness_laser']:.2f} / {rep['icp_fitness_polycam']:.2f}" for rep in reports)]
    out = ROOT / "bench/results"
    (out / "head_to_head.json").write_text(json.dumps(reports, indent=1, default=float))
    (out / "head_to_head.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
