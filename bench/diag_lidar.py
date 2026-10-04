"""D4 + D5: re-run the LiDAR pipeline with/without the depth-bias correction and with drift on/off,
score each against the Faro laser, and list every laser surface near each scored wall (to catch the
reference matching the wrong one of two parallel surfaces).

    python bench/diag_lidar.py <scene_dir> --affine A B [--tag name]
A, B come from `python -m roomscope.eval.depth_bias` on a *different* capture (hold-out).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from roomscope.eval.laser import score_capture
from roomscope.pipeline import run_capture


def wall_surfaces(result: dict, ref, win: float = 0.25) -> list[dict]:
    """For each polygon edge: laser surface peaks (1 cm bins, facing inward, mid-height) within +-win."""
    L, LN = np.asarray(ref.points), np.asarray(ref.normals)
    fz = np.percentile(L[LN[:, 2] > 0.95, 2], 5)
    rows = []
    for room in result["rooms"]:
        poly = room["polygon"]
        for k in range(len(poly)):
            p, q = np.array(poly[k]), np.array(poly[(k + 1) % len(poly)])
            t = (q - p) / np.linalg.norm(q - p)
            axis = 0 if abs(t[0]) < abs(t[1]) else 1
            inward = np.array([-t[1], t[0], 0.0])
            c = p[axis]
            lo, hi = sorted((p[1 - axis], q[1 - axis]))
            m = (LN @ inward > np.cos(np.deg2rad(15))) & (np.abs(L[:, axis] - c) < win)
            m &= (L[:, 1 - axis] > lo + 0.1) & (L[:, 1 - axis] < hi - 0.1)
            m &= (L[:, 2] > fz + 0.3) & (L[:, 2] < fz + 2.0)
            h, e = np.histogram(L[m, axis], np.arange(c - win, c + win + 0.01, 0.01))
            pk = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 0.15 * h.max()]
            pk = sorted(pk, key=lambda i: -h[i])[:3]
            rows.append({"wall": f"{room['id']}-W{k + 1}", "len_m": round(float(hi - lo), 2),
                         "surfaces": [(round(float(e[i] + 0.005 - c) * 100, 1), int(h[i])) for i in pk]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("--affine", type=float, nargs=2, required=True)
    ap.add_argument("--tag", default="diag")
    a = ap.parse_args()
    out = Path("out/diag") / a.tag
    summary = {}
    for corr in (False, True):
        for drift in (False, True):
            name = f"{'corr' if corr else 'raw'}_{'drift' if drift else 'nodrift'}"
            kw = {"depth_affine": tuple(a.affine)} if corr else None
            run_capture(a.scene, "lidar", out / name, drift=drift, load_kw=kw, depth_correction=corr)
            sc, ref = score_capture(out / name / a.scene.name, a.scene, return_ref=True)
            res = json.loads((out / name / a.scene.name / "result.json").read_text())
            ce = [r["err"] for r in sc["ceil"]]
            pe = np.array([r["offset_err"] for r in sc["wall_planes"]])
            le = np.array([r["err"] for r in sc["walls"]])
            summary[name] = {"ceiling_err_cm": [round(x * 100, 2) for x in ce],
                             "plane_err_cm": [round(x * 100, 1) for x in pe],
                             "plane_MAE_cm": round(float(np.mean(np.abs(pe))) * 100, 2),
                             "plane_median_cm": round(float(np.median(np.abs(pe))) * 100, 2),
                             "length_MAE_cm": round(float(np.mean(np.abs(le))) * 100, 2) if len(le) else None,
                             "drift": sc["drift"]}
            print(f"\n=== {name}: ceiling {summary[name]['ceiling_err_cm']} cm | wall planes MAE "
                  f"{summary[name]['plane_MAE_cm']} median {summary[name]['plane_median_cm']} cm | "
                  f"lengths MAE {summary[name]['length_MAE_cm']} cm")
            print("   per-wall plane err (cm):", summary[name]["plane_err_cm"])
            if name == "raw_nodrift":
                print("   laser surfaces near each wall (offset cm from our wall, count):")
                for r in wall_surfaces(res, ref):
                    print("    ", r)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
