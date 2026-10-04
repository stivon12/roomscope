"""D1/D6: is the phone LiDAR depth itself biased? Pose-free per-pixel comparison.

ARKitScenes `highres_depth` (upsampling subset) is depth rendered from the Faro laser scan at the
laser-registered camera pose of that exact frame. Sampling it at the 256x192 `lowres_depth` pixel
centres gives true depth and measured depth for the same ray, so ARKit pose error plays no part.

Error is binned by range, incidence angle, confidence and image radius:
- error proportional to range  -> range-scale bias (H1);
- error growing with image radius but flat in range -> intrinsics / distortion;
- error growing with incidence angle -> grazing-angle bias.

    python -m roomscope.eval.depth_bias <scene_dir> [<scene_dir> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


def _pincam(root: Path, vid: str, tstr: str) -> np.ndarray | None:
    t = float(tstr)
    for c in (tstr, f"{t - 0.001:.3f}", f"{t + 0.001:.3f}"):
        p = root / "lowres_wide_intrinsics" / f"{vid}_{c}.pincam"
        if p.exists():
            return np.loadtxt(p)
    return None


def _normals(z: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Per-pixel unit normals (camera frame) of a depth image, from central differences."""
    H, W = z.shape
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    P = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], -1)
    dx = np.zeros_like(P); dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1] = P[2:] - P[:-2]
    n = np.cross(dx, dy)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)


def collect(scene: Path) -> dict[str, np.ndarray]:
    """One row per usable lowres pixel: true depth, measured depth, confidence, incidence, radius, frame."""
    scene = Path(scene)
    rows = {k: [] for k in ("gt", "meas", "conf", "inc", "rad", "frame", "vert")}
    from .pose_drift import _pose_at
    traj = np.loadtxt(scene / "lowres_wide.traj")
    files = sorted((scene / "highres_depth").glob("*.png"))
    for fi, f in enumerate(files):
        vid, tstr = f.stem.rsplit("_", 1)
        lo_p, cf_p = scene / "lowres_depth" / f.name, scene / "confidence" / f.name
        pin = _pincam(scene, vid, tstr)
        if pin is None or not lo_p.exists() or not cf_p.exists():
            continue
        w, h, fx, fy, cx, cy = pin
        lo = cv2.imread(str(lo_p), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        cf = cv2.imread(str(cf_p), cv2.IMREAD_UNCHANGED)
        hi = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        H, W = lo.shape
        sx, sy = hi.shape[1] / W, hi.shape[0] / H
        # true depth at each lowres pixel centre: median of the 3x3 highres neighbourhood, and only
        # where that neighbourhood is smooth (no depth edge, no hole) so the comparison is like-for-like
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        U = np.clip(((u + 0.5) * sx).astype(int), 1, hi.shape[1] - 2)
        V = np.clip(((v + 0.5) * sy).astype(int), 1, hi.shape[0] - 2)
        nb = np.stack([hi[V + dv, U + du] for dv in (-1, 0, 1) for du in (-1, 0, 1)], 0)
        gt = np.median(nb, 0)
        smooth = (nb.min(0) > 0.1) & (nb.max(0) - nb.min(0) < 0.01 * gt + 0.005)
        n = _normals(np.where(gt > 0, gt, np.nan), fx, fy, cx, cy)
        ray = np.stack([(u + 0.5 - cx) / fx, (v + 0.5 - cy) / fy, np.ones_like(gt)], -1)
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        inc = np.degrees(np.arccos(np.clip(np.abs(np.sum(n * ray, -1)), 0, 1)))
        rad = np.hypot((u + 0.5 - cx) / (W / 2), (v + 0.5 - cy) / (H / 2))   # 0 centre .. ~1.4 corner
        ok = smooth & (lo > 0.1) & np.isfinite(inc) & (gt < 6.0)
        # surface orientation in the gravity-aligned ARKit world (y up): |n.y| -> 1 floor/ceiling, -> 0 wall
        E = _pose_at(traj, float(tstr))
        if E is None:
            continue
        ny = np.abs(n @ E[:3, :3][:, 1])              # world y-axis expressed in camera coords
        vert = np.where(ny > 0.9, 0, np.where(ny < 0.2, 1, 2))   # 0 horizontal, 1 wall, 2 oblique
        for k, a in (("gt", gt), ("meas", lo), ("conf", cf), ("inc", inc), ("rad", rad), ("vert", vert)):
            rows[k].append(a[ok])
        rows["frame"].append(np.full(int(ok.sum()), fi))
    return {k: np.concatenate(v) for k, v in rows.items()}


def _table(name: str, x: np.ndarray, rel: np.ndarray, err: np.ndarray, edges) -> str:
    out = [f"\n{name:>14} {'n':>9} {'median err':>11} {'median rel':>11} {'MAD rel':>8}"]
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() < 500:
            continue
        r = rel[m]
        out.append(f"{f'[{a:g},{b:g})':>14} {m.sum():>9d} {np.median(err[m]) * 100:>9.2f}cm "
                   f"{np.median(r) * 100:>10.2f}% {np.median(np.abs(r - np.median(r))) * 100:>7.2f}%")
    return "\n".join(out)


def report(scene: Path) -> dict:
    d = collect(scene)
    err = d["meas"] - d["gt"]
    rel = err / d["gt"]
    hc = d["conf"] == 2
    lines = [f"scene {Path(scene).name}: {len(err):,} smooth pixel pairs from "
             f"{len(np.unique(d['frame']))} frames ({hc.mean() * 100:.0f}% confidence 2)"]
    lines.append(_table("conf", d["conf"].astype(float), rel, err, [0, 1, 2, 3]))
    e, r = err[hc], rel[hc]
    lines.append("\n-- confidence 2 only (what the pipeline uses) --")
    lines.append(_table("range m", d["gt"][hc], r, e, [0.3, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]))
    lines.append(_table("incidence deg", d["inc"][hc], r, e, [0, 15, 30, 45, 60, 75, 90]))
    lines.append(_table("image radius", d["rad"][hc], r, e, [0, 0.25, 0.5, 0.75, 1.0, 1.5]))
    lines.append("\nsurface: 0 = floor/ceiling, 1 = wall, 2 = oblique")
    lines.append(_table("surface", d["vert"][hc].astype(float), r, e, [0, 1, 2, 3]))
    for sv, nm in ((0, "floor/ceiling"), (1, "wall")):
        ms = hc & (d["vert"] == sv)
        lines.append(_table(f"{nm} range", d["gt"][ms], rel[ms], err[ms], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]))
    # robust linear model meas = a*gt + b on confidence-2 pixels with incidence < 60 deg
    m = hc & (d["inc"] < 60)
    g, y = d["gt"][m], d["meas"][m]
    keep = np.ones(len(g), bool)
    for _ in range(5):
        A = np.c_[g[keep], np.ones(keep.sum())]
        a, b = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        res = y - (a * g + b)
        keep = np.abs(res) < 3 * 1.4826 * np.median(np.abs(res[keep]))
    # per-frame median relative error: spread tells how much is per-frame registration noise (H4)
    fr = np.array([np.median(rel[hc & (d["frame"] == f)]) for f in np.unique(d["frame"][hc])
                   if (hc & (d["frame"] == f)).sum() > 300])
    k_med = float(np.median(rel[m]))
    lines.append(f"\nfit (conf 2, incidence<60): meas = {a:.4f} * true {b * 100:+.2f} cm  "
                 f"-> scale {(a - 1) * 100:+.2f}%  | median rel err {k_med * 100:+.2f}%")
    lines.append(f"per-frame median rel err: median {np.median(fr) * 100:+.2f}%  "
                 f"IQR [{np.percentile(fr, 25) * 100:+.2f}, {np.percentile(fr, 75) * 100:+.2f}]%  "
                 f"n_frames={len(fr)}")
    txt = "\n".join(lines)
    return {"text": txt, "scale": float(a), "offset": float(b), "median_rel": k_med,
            "frame_rel": fr}


if __name__ == "__main__":
    outdir = Path("out/diag")
    outdir.mkdir(parents=True, exist_ok=True)
    for s in sys.argv[1:]:
        r = report(Path(s))
        print(r["text"])
        (outdir / f"depth_bias_{Path(s).name}.txt").write_text(r["text"] + "\n")
