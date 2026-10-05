"""How well does a monocular metric depth model set absolute scale? Checked against LiDAR, frame by frame.

    python bench/scale_cues.py <scene_dir> [--n 24]

For n frames of an ARKitScenes .mov: frame i has the timestamp of .pincam i (same count as video frames),
which names the matching lowres_depth / confidence PNGs. DA3METRIC-LARGE (frontends/da3_worker.py, run in
a subprocess) predicts metric depth with the frame's own focal length; its median ratio to the LiDAR depth
(confidence 2, corrected by the held-out depth scale) over the frame's pixels is that frame's scale
error. Reported: median across frames and the spread, i.e. how far one video's median would be off.
Only frames stored in sensor orientation (not auto-rotated by the decoder) are used, so pixels align.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from roomscope.core.depth_calib import resolve_depth_scale

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    scene = a.scene
    vid = next(scene.glob("*.mov"))
    pins = sorted((scene / "lowres_wide_intrinsics").glob("*.pincam"))
    cap = cv2.VideoCapture(str(vid))
    total, W, H = int(cap.get(7)), int(cap.get(3)), int(cap.get(4))
    assert len(pins) == total, "pincam count must equal the frame count"
    if H > W:
        raise SystemExit("video decodes rotated; pixel alignment with lowres_depth not handled here")
    ds = resolve_depth_scale().scale
    want = set(np.linspace(int(0.05 * total), int(0.95 * total), a.n).astype(int).tolist())
    tmp = Path(tempfile.mkdtemp())
    jobs, lidar = [], []
    for i in range(total):
        if not cap.grab():
            break
        if i not in want:
            continue
        ok, f = cap.retrieve()
        stem = pins[i].stem
        dp, cp = scene / "lowres_depth" / f"{stem}.png", scene / "confidence" / f"{stem}.png"
        if not (ok and dp.exists() and cp.exists()):
            continue
        w, h, fx, fy, cx, cy = np.loadtxt(pins[i])
        p = tmp / f"{i:06d}.jpg"
        cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
        sc = W / w
        jobs.append((str(p), [[fx * sc, 0, cx * sc], [0, fy * sc, cy * sc], [0, 0, 1]]))
        d = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0 / ds
        c = cv2.imread(str(cp), cv2.IMREAD_UNCHANGED)
        lidar.append(np.where((c >= 2) & (d > 0.3) & (d < 4.0), d, np.nan))
    cap.release()
    (tmp / "jobs.json").write_text(json.dumps({"mode": "mono_metric", "images": [j[0] for j in jobs],
                                               "K": [j[1] for j in jobs]}))
    subprocess.run([sys.executable, "-m", "roomscope.frontends.da3_worker",
                    str(tmp / "jobs.json"), str(tmp / "da3.npz")], check=True)
    z = np.load(tmp / "da3.npz")
    ratios = []
    for i, L in enumerate(lidar):
        D = cv2.resize(z[f"depth_{i}"], (L.shape[1], L.shape[0]), interpolation=cv2.INTER_AREA)
        m = np.isfinite(L) & (D > 0)
        if m.sum() > 500:
            ratios.append(float(np.median(D[m] / L[m])))
    r = np.array(ratios)
    print(json.dumps({"capture": scene.name, "frames": len(r), "da3_over_lidar_median": round(float(np.median(r)), 4),
                      "scale_err_pct": round((float(np.median(r)) - 1) * 100, 2),
                      "per_frame_iqr_pct": [round((float(np.percentile(r, q)) - 1) * 100, 1) for q in (25, 75)],
                      "per_frame": [round(x, 3) for x in r]}))


if __name__ == "__main__":
    main()
