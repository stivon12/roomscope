"""Do semantic labels separate walls from furniture, curtains and doors? Per candidate wall plane: what the
current geometry decided and what the labels of its points say.

    python bench/semantic_audit.py <capture> [<capture> ...] [--interval 1.0]

For each capture: fuse the LiDAR frames (no drift correction), label the cloud (core/semantics.py), then
run the candidate search of core/layout.fit_wall_planes with its own thresholds and record each candidate's
fate (too low / seen over -> furniture / builtin / wall / dropped by drop_layered) next to the label
shares of its points. Written to out/semantic_audit/<capture>.json and printed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from roomscope.core import layout as L
from roomscope.core.semantics import UNKNOWN, label_cloud, label_frames
from roomscope.frontends.lidar import load_any


def candidates(cloud, floor, ceil_z, sem, names, min_top=1.0, max_seen_over=0.5):
    out = []
    zrel = cloud.P[:, 2] - floor.z(cloud.P[:, 0], cloud.P[:, 1])
    for face, (axis, sign) in L.FACES.items():
        m = (cloud.cls == face) & (zrel > 0.05) & (zrel < ceil_z - 0.05)
        Q, zq, sq = cloud.P[m], zrel[m], sem[m]
        if len(Q) < 100:
            continue
        x = Q[:, axis]
        edges = np.arange(x.min() - 0.02, x.max() + 0.03, 0.01)
        hs = gaussian_filter1d(np.histogram(x, edges)[0].astype(float), 1.5)
        for pk in L._peaks(hs, min_count=max(60, 0.002 * len(Q)), min_sep=8):
            off, win = edges[pk] + 0.005, 0.04
            for _ in range(4):
                sel = np.abs(x - off) < win
                if sel.sum() < 30:
                    break
                off = float(np.mean(x[sel]))
                win = max(2.5 * np.std(x[sel] - off), 0.012)
            sel = np.abs(x - off) < win
            if sel.sum() < 60:
                continue
            top = float(np.percentile(zq[sel], 97))
            segs = L._segments(Q[sel, 1 - axis], 1.25)
            lab = sq[sel]
            known = lab[lab != UNKNOWN]
            u, c = np.unique(known, return_counts=True)
            shares = {names[k].strip(): round(float(n) / max(len(known), 1), 3)
                      for k, n in sorted(zip(u, c), key=lambda t: -t[1])[:4]}
            rec = {"face": face, "offset": round(off, 3), "n": int(sel.sum()), "top": round(top, 2),
                   "std_cm": round(100 * float(np.std(x[sel] - off)), 1), "length": round(sum(h - l for l, h in segs), 2),
                   "labelled": round(len(known) / len(lab), 2), "labels": shares}
            if top < min_top:
                rec["fate"] = "rejected: too low"
            elif not segs:
                rec["fate"] = "rejected: no segment"
            else:
                so = L.seen_over_fraction(cloud, floor, axis, sign, off, segs, Q[sel, 1 - axis], zq[sel], ceil_z)
                rec["seen_over"] = round(so, 2)
                rec["fate"] = ("rejected: seen over" if top < 1.8 else "builtin") if so > max_seen_over else "wall"
                rec["_plane"] = L.WallPlane(face, axis, sign, off, float(np.std(x[sel] - off)), int(sel.sum()),
                                            segs, top, "builtin" if rec["fate"] == "builtin" else "wall")
            out.append(rec)
    kept = [r["_plane"] for r in out if "_plane" in r]
    after = {(p.face, round(p.offset, 3)) for p in L.drop_layered(kept, ceil_z)}
    for r in out:
        if "_plane" in r:
            if (r["face"], r["offset"]) not in after:
                r["fate"] = "dropped: layered in front of a wall"
            del r["_plane"]
    return out


def audit(capture: Path, interval: float) -> dict:
    cap = load_any(capture)
    cloud = L.fuse(cap, None, frame_step=1 if len(cap.poses) <= 100 else 2)
    fp = cloud.P[cloud.cls == "floor"]
    floor = L.fit_hplane(fp, z0=L.pick_level(fp, "low"))
    cp = cloud.P[cloud.cls == "ceil"]
    ccfg = L.layout_config()["ceiling"]
    ceil_z = (L.pick_level(cp, "high") - floor.c) if len(cp) > 100 else ccfg["max_m"]
    frames = label_frames(cap, Path("out/semantic_audit/cache"), interval)
    sem, _ = label_cloud(cloud, cap, None, frames)
    names = label_frames.names
    return {"capture": str(capture), "frames_labelled": len(frames), "ceil_z": round(ceil_z, 2),
            "cloud_labelled": round(float(np.mean(sem != UNKNOWN)), 3),
            "candidates": candidates(cloud, floor, ceil_z, sem, names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+", type=Path)
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()
    out = Path("out/semantic_audit")
    out.mkdir(parents=True, exist_ok=True)
    for c in a.captures:
        r = audit(c, a.interval)
        (out / f"{c.name}.json").write_text(json.dumps(r, indent=1))
        print(f"== {c.name}: {r['frames_labelled']} frames labelled, {100 * r['cloud_labelled']:.0f}% of cloud labelled")
        for x in r["candidates"]:
            print(f"  {x['face']:>2} {x['offset']:7.3f}  len {x['length']:4.1f} top {x['top']:4.2f} std {x['std_cm']:4.1f}  "
                  f"{x['fate']:<36} {x['labels']}")


if __name__ == "__main__":
    main()
