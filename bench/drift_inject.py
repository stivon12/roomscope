"""Known-answer drift test on real captures: inject a slow drift into the real ARKit poses, run the
drift correction, and measure how much of the injected drift it removes.

ARKit's own poses are good to about 2 cm on these single-room captures (eval/pose_drift.py), so they
serve as the reference. Injected drift grows linearly over the capture to `trans` (m) and `yaw_deg`,
the kind of slow VIO drift a multi-room walk accumulates. Error is the camera-position difference from
the original poses after removing the capture-wide mean offset (the gauge).

    python bench/drift_inject.py <scene_dir> [--trans 0.15 0.10 0.03] [--yaw 1.5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from roomscope.core import drift as D
from roomscope.core.depth_calib import resolve_depth_scale
from roomscope.core.layout import rz
from roomscope.frontends.lidar import load_any


def inject(cap, trans, yaw_deg):
    s = (cap.timestamps - cap.timestamps[0]) / np.ptp(cap.timestamps)
    piv = cap.poses[:, :3, 3].mean(0)
    out = cap.poses.copy()
    for i, a in enumerate(s):
        M = rz(np.deg2rad(yaw_deg) * a)
        M[:3, 3] = piv - M[:3, :3] @ piv + np.asarray(trans) * a
        out[i] = M @ cap.poses[i]
    return out


def cam_err(poses, ref):
    d = poses[:, :3, 3] - ref[:, :3, 3]
    d -= np.median(d, 0)
    m = np.linalg.norm(d, axis=1)
    return np.median(m) * 100, np.percentile(m, 90) * 100, m.max() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("--trans", type=float, nargs=3, default=[0.15, 0.10, 0.03])
    ap.add_argument("--yaw", type=float, default=1.5)
    ap.add_argument("--only", default="", help="comma-separated variant names")
    a = ap.parse_args()
    ds = resolve_depth_scale()
    cap = load_any(a.scene, depth_affine=(ds.scale, 0.0))
    ref = cap.poses.copy()
    cap.poses = inject(cap, a.trans, a.yaw)
    print(f"{a.scene.name}: injected drift {a.trans} m, {a.yaw} deg yaw over {np.ptp(cap.timestamps):.0f} s")
    print(f"  {'no correction':34s} median {cam_err(cap.poses, ref)[0]:5.1f} cm  p90 {cam_err(cap.poses, ref)[1]:5.1f}  "
          f"max {cam_err(cap.poses, ref)[2]:5.1f}")
    variants = {
        "per-fragment yaw, 20cm (old default)": dict(),
        "no yaw, 8cm, merge 6cm, prior 3cm": dict(per_fragment_yaw=False, match_dist=0.08, merge_dist=0.06,
                                                  reassoc_iters=4, prior_sigma=0.03),
        "no yaw, 8cm, prior 3cm": dict(per_fragment_yaw=False, match_dist=0.08, prior_sigma=0.03),
        "no yaw, 8cm, prior 1cm": dict(per_fragment_yaw=False, match_dist=0.08, prior_sigma=0.01),
        "yaw, 8cm, prior 3cm": dict(per_fragment_yaw=True, match_dist=0.08, prior_sigma=0.03),
        "no yaw, 8cm, merge 6cm": dict(per_fragment_yaw=False, match_dist=0.08, merge_dist=0.06, reassoc_iters=4),
        "gated linear yaw, 8cm, merge 6cm": dict(per_fragment_yaw="linear", match_dist=0.08, merge_dist=0.06, reassoc_iters=4),
        "linear yaw, 8cm, merge 6cm, prior 3cm": dict(per_fragment_yaw="linear", match_dist=0.08, merge_dist=0.06,
                                                     reassoc_iters=4, prior_sigma=0.03),
        "default": dict(),
        "structural": dict(structural=True),
        "structural + cauchy": dict(structural=True, loss="cauchy"),
        "structural + merge 6cm": dict(structural=True, merge_dist=0.06, reassoc_iters=4),
        "structural + cauchy + merge 6cm": dict(structural=True, loss="cauchy", merge_dist=0.06, reassoc_iters=4),
    }
    if a.only:
        variants = {k: v for k, v in variants.items() if k in a.only.split(",")}
    for name, kw in variants.items():
        C = D.correct_drift(cap, **kw).corrections
        corrected = np.einsum("fij,fjk->fik", C, cap.poses)
        e = cam_err(corrected, ref)
        print(f"  {name:34s} median {e[0]:5.1f} cm  p90 {e[1]:5.1f}  max {e[2]:5.1f}", flush=True)


if __name__ == "__main__":
    main()
