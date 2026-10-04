"""Score drift-correction variants on what they are meant to fix: per-frame camera position against
the Faro laser (eval/pose_drift.py method; laser-rendered frames, so phone depth error plays no part).

    python bench/drift_eval.py <scene_dir> <nodrift_out_dir>
<nodrift_out_dir> is a --no-drift pipeline output for the scene (its cloud registers the laser).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from roomscope.core import drift as D
from roomscope.core.depth_calib import resolve_depth_scale
from roomscope.eval.pose_drift import frame_errors, laser_in_result, summarise_errors
from roomscope.frontends.lidar import load_any


def main(scene: Path, nodrift_out: Path):
    ds = resolve_depth_scale()
    cap = load_any(scene, depth_affine=(ds.scale, 0.0))
    ref, T_res = laser_in_result(scene, nodrift_out)
    variants = {
        "none (ARKit as-is)": None,
        "current (yaw + 20cm match)": dict(),
        "no per-fragment yaw": dict(per_fragment_yaw=False),
        "8cm match": dict(match_dist=0.08),
        "no yaw + 8cm match": dict(per_fragment_yaw=False, match_dist=0.08),
        "no yaw + 8cm + prior 3cm": dict(per_fragment_yaw=False, match_dist=0.08, prior_sigma=0.03),
        "no yaw + 8cm + prior 1cm": dict(per_fragment_yaw=False, match_dist=0.08, prior_sigma=0.01),
        "no yaw + 8cm + prior 0.5cm": dict(per_fragment_yaw=False, match_dist=0.08, prior_sigma=0.005),
        "linear yaw + 8cm + merge 6cm": dict(per_fragment_yaw="linear", match_dist=0.08, merge_dist=0.06,
                                             reassoc_iters=4),
        "default": dict(),
        "structural": dict(structural=True),
        "structural + cauchy": dict(structural=True, loss="cauchy"),
        "structural + merge 6cm": dict(structural=True, merge_dist=0.06, reassoc_iters=4),
        "structural + cauchy + merge 6cm": dict(structural=True, loss="cauchy", merge_dist=0.06, reassoc_iters=4),
    }
    if len(sys.argv) > 3:
        variants = {k: v for k, v in variants.items() if k in sys.argv[3].split(",")}
    results, good = {}, None
    for name, kw in variants.items():
        if kw is None:
            fn = None
        else:
            corr = D.correct_drift(cap, **kw).corrections
            fn = (lambda C: (lambda t: C[int(np.argmin(np.abs(cap.timestamps - t)))]))(corr)
        A = frame_errors(scene, ref, T_res, corr_fn=fn, G=cap.align, step=2)
        if good is None:                    # same frame set for every variant
            good = (A[:, 5] > 0.6) & (A[:, 7] > 0.05)
            t_good = A[good, 0]
        g = np.isin(A[:, 0], t_good)
        results[name] = summarise_errors(A, g)
        r = results[name]
        print(f"{name:30s} camera pos err: median {r['median_cm']:.1f} cm  p90 {r['p90_cm']:.1f}  "
              f"max {r['max_cm']:.1f}  | rot median {r['rot_median_deg']:.2f} deg  (n={g.sum()})", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
