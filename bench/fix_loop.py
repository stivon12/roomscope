"""Fix loop, regenerable before/after: the LiDAR depth-bias correction (fix/DECLARATION.md).

For each capture of room 421337, the current pipeline runs twice, identical except for the fix:
  before = --no-depth-correction
  after  = depth divided by a scale fitted on the OTHER two captures only (leave-one-capture-out, so no
           capture is scored with a correction it helped fit).
Both are scored against the Faro laser; results go to fix/before_after.{json,md}.

    python bench/fix_loop.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from roomscope.eval.laser import score_capture
from roomscope.pipeline import run_capture

ROOT = Path("benchmark/raw/arkitscenes/raw/Validation")
# per-capture median relative depth error, conf 2, vs laser-rendered depth (out/diag/depth_bias_*.txt,
# python -m roomscope.eval.depth_bias)
REL_ERR = {"42444946": -0.0130, "42444949": -0.0110, "42444950": -0.0123}


def held_out_scale(cid: str) -> float:
    others = [v for k, v in REL_ERR.items() if k != cid]
    return 1.0 + float(np.median(others))


def score(out: Path, scene: Path) -> dict:
    sc = score_capture(out, scene)
    ce = [r["err"] for r in sc["ceil"]]
    pe = np.array([r["offset_err"] for r in sc["wall_planes"]])
    le = np.array([r["err"] for r in sc["walls"]])
    return {"ceiling_err_cm": round(ce[0] * 100, 2) if ce else None,
            "wall_plane_mae_cm": round(float(np.mean(np.abs(pe))) * 100, 2) if len(pe) else None,
            "wall_plane_median_cm": round(float(np.median(np.abs(pe))) * 100, 2) if len(pe) else None,
            "wall_length_mae_cm": round(float(np.mean(np.abs(le))) * 100, 2) if len(le) else None}


def main():
    rows = {}
    for cid in REL_ERR:
        scene = ROOT / cid
        out = Path("out/fix") / cid
        s = held_out_scale(cid)
        run_capture(scene, "lidar", out / "before", depth_correction=False, calibrate=False)
        run_capture(scene, "lidar", out / "after", depth_scale=s, calibrate=False)
        rows[cid] = {"held_out_scale": round(s, 5),
                     "before": score(out / "before" / cid, scene),
                     "after": score(out / "after" / cid, scene)}
        print(cid, json.dumps(rows[cid]), flush=True)
    Path("fix").mkdir(exist_ok=True)
    Path("fix/before_after.json").write_text(json.dumps(rows, indent=2))
    lines = ["| capture | held-out scale | ceiling err cm (before -> after) | wall plane MAE cm | wall length MAE cm |",
             "|---|---|---|---|---|"]
    for cid, r in rows.items():
        b, a = r["before"], r["after"]
        lines.append(f"| {cid} | {r['held_out_scale']} | {b['ceiling_err_cm']} -> {a['ceiling_err_cm']} | "
                     f"{b['wall_plane_mae_cm']} -> {a['wall_plane_mae_cm']} | "
                     f"{b['wall_length_mae_cm']} -> {a['wall_length_mae_cm']} |")
    Path("fix/before_after.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
