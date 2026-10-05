"""Damage regions on the room's surfaces, from per-frame candidates voted across views.

1. Keyframes (about one per `interval_s`) go to the detector subprocess (damage/worker.py): box proposals,
   each classified against damage and clean-surface descriptions (config/damage.yaml), one mask per kept box.
2. Every wall and ceiling of the result is a grid of `cell_m` cells in the result frame. Each cell is projected
   into every keyframe with that frame's pose. It counts as seen when it is in the image, in front of the camera
   within `max_range_m`, not too oblique (`max_angle_deg`) and not hidden: the frame's own LiDAR depth at that
   pixel is not more than `occl_m` in front of the cell. A seen cell counts as positive for a class when it
   falls inside that class's mask.
3. A cell is damaged when at least `min_views` frames call it positive and they are at least `min_ratio` of the
   frames that saw it. Damage that is only in one view, or that moves between views (reflections in mirrors and
   windows, a shadow, a person), does not survive. Connected cells form a region.
4. Area comes from the cells; its range from stricter / looser agreement (`ratio_lo` / `ratio_hi`), so a region
   seen consistently has a tight range and a disputed one a wide range. Intervals are NOT conformally
   calibrated (no labelled damage captures yet): method says so.

Only the LiDAR tier carries the per-frame depth this needs; other tiers report no damage with a warning.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy import ndimage
from shapely import contains_xy
from shapely.geometry import Polygon

from ..measure import Measurement

ROOT = Path(__file__).resolve().parents[3]


def damage_config() -> dict:
    return yaml.safe_load((ROOT / "config" / "damage.yaml").read_text())


def _run_worker(job: dict, cache_dir: Path) -> dict:
    key = hashlib.sha1(json.dumps(job, sort_keys=True).encode()
                       + b"".join(Path(p).read_bytes()[:4096] + str(Path(p).stat().st_size).encode()
                                  for p in job["images"])).hexdigest()[:12]
    cp = Path(cache_dir) / f"damage_{key}.npz"
    if not cp.exists():
        with tempfile.TemporaryDirectory() as td:
            jp, op = Path(td) / "job.json", Path(td) / "out.npz"
            jp.write_text(json.dumps(job))
            subprocess.run([sys.executable, "-m", "roomscope.damage.worker", str(jp), str(op)], check=True,
                           env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(op.read_bytes())
    z = np.load(cp)
    return {k: z[k] for k in z.files}


def _surfaces(result: dict, floor_z, cell: float) -> list[dict]:
    """Wall and ceiling grids in the result frame. Walls: u along the wall from its start, v up from the floor.
    Ceiling: u = x, v = y over the room polygon's bounding box (cells outside the polygon are dropped)."""
    out = []
    for room in result["rooms"]:
        h = room["ceiling_height"]["value"]
        for w in room["walls"]:
            p, q = np.array(w["start"], float), np.array(w["end"], float)
            L = float(np.linalg.norm(q - p))
            if L < 2 * cell:
                continue
            t = (q - p) / L
            nu, nv = int(L / cell), int(h / cell)
            uu, vv = np.meshgrid((np.arange(nu) + 0.5) * cell, (np.arange(nv) + 0.5) * cell)
            xy = p + uu[..., None] * t
            z = floor_z(xy[..., 0], xy[..., 1]) + vv
            out.append({"id": f"S-{w['id']}", "kind": "wall", "room": room["id"], "C": np.dstack([xy, z]),
                        "n": np.array([-t[1], t[0], 0.0]), "valid": np.ones((nv, nu), bool), "cell": cell})
        poly = Polygon(room["polygon"])
        x0, y0, x1, y1 = poly.bounds
        nu, nv = int((x1 - x0) / cell), int((y1 - y0) / cell)
        xx, yy = np.meshgrid(x0 + (np.arange(nu) + 0.5) * cell, y0 + (np.arange(nv) + 0.5) * cell)
        valid = contains_xy(poly.buffer(-cell), xx, yy)
        z = floor_z(xx, yy) + h
        out.append({"id": f"S-{room['id']}-ceiling", "kind": "ceiling", "room": room["id"],
                    "C": np.dstack([xx, yy, z]), "n": np.array([0.0, 0.0, -1.0]), "valid": valid, "cell": cell,
                    "origin": (x0, y0)})
    return out


def _depth_image(P: np.ndarray, K: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Nearest LiDAR depth per depth pixel from the frame's own points (camera looks along -z)."""
    H, W = hw
    z = -P[:, 2]
    u = np.floor(K[0, 0] * P[:, 0] / z + K[0, 2]).astype(int)
    v = np.floor(-K[1, 1] * P[:, 1] / z + K[1, 2]).astype(int)
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    D = np.full((H, W), np.inf, np.float32)
    np.minimum.at(D, (v[ok], u[ok]), z[ok])
    return D


def detect(cap, frame_to_result, result: dict, floor_z, cache_dir: Path, warnings: list[str]) -> tuple[list[dict], dict]:
    """Damage regions (schema `damage_regions`) for a LiDAR capture, plus the model record for meta.models."""
    from ..core.semantics import _video_frames, pick_frames, upright_k

    cfg = damage_config()
    if getattr(cap, "depth_hw", None) is None:
        warnings.append("damage detection needs per-frame depth (LiDAR tier); not run on this tier")
        return [], {}
    names = list(cfg["classes"])
    sel = pick_frames(cap.timestamps, cfg["interval_s"])
    vid, vframe = _video_frames(cap)
    Hd, Wd = cap.depth_hw
    with tempfile.TemporaryDirectory() as td:
        v = cv2.VideoCapture(str(vid))
        paths, dark, keep = [], 0, []
        for i in sel:
            v.set(cv2.CAP_PROP_POS_FRAMES, int(vframe[i]))
            ok, im = v.read()
            if not ok:
                continue
            if cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean() < cfg["min_luma"]:
                dark += 1                         # too dark to judge: insufficient evidence, not "no damage"
                continue
            p = Path(td) / f"{i:06d}.jpg"
            cv2.imwrite(str(p), im)
            paths.append(str(p))
            keep.append(i)
        rgb_hw = (int(v.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(v.get(cv2.CAP_PROP_FRAME_WIDTH)))
        v.release()
        if dark:
            warnings.append(f"damage: {dark} of {len(sel)} keyframes too dark to judge (insufficient evidence)")
        if not paths:
            return [], {}
        job = {"images": paths, "rot90": [upright_k(cap.poses[i]) for i in keep], "classes": cfg["classes"],
               "describe": cfg["describe"], "negatives": cfg["negatives"], "class_threshold": cfg["class_threshold"],
               "box_threshold": cfg["box_threshold"], "text_threshold": cfg["text_threshold"],
               "work_px": cfg["work_px"], "max_box_frac": cfg["max_box_frac"]}
        z = _run_worker(job, cache_dir)
    model = {"name": "Grounding DINO + SigLIP 2 + SAM 2.1", "version": str(z["model"]), "license": "Apache-2.0",
             "use": "damage proposals, crop classification against damage and clean-surface labels, masks; "
                    "voted across views on the surfaces"}

    cell = cfg["cell_m"]
    surfs = _surfaces(result, floor_z, cell)
    for s in surfs:
        sh = s["valid"].shape
        s["seen"] = np.zeros(sh, np.int32)
        s["pos"] = np.zeros((len(names),) + sh, np.int32)
        s["score"] = np.zeros((len(names),) + sh, np.float32)
        s["frames"] = [[] for _ in names]
    cos_max = np.cos(np.deg2rad(cfg["max_angle_deg"]))
    for j, i in enumerate(keep):
        dets, masks = z[f"det_{j}"], z[f"mask_{j}"]
        mh, mw = (int(a) for a in z[f"hw_{j}"])
        cls_mask = np.zeros((len(names), mh, mw), np.float32)       # best score of each class per pixel
        for d, m in zip(dets, masks):
            ci = int(d[0])
            if cfg.get("dilate_px", {}).get(names[ci]):
                r = cfg["dilate_px"][names[ci]]
                m = cv2.dilate(m.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
            cls_mask[ci] = np.maximum(cls_mask[ci], m * d[1])
        T = frame_to_result(i) @ cap.poses[i]                      # camera -> result
        Ti = np.linalg.inv(T)
        cam = T[:3, 3]
        D = _depth_image(cap.pts_cam[i], cap.K_depth, (Hd, Wd))
        Km = cap.K_depth.copy()
        Km[0] *= mw / Wd
        Km[1] *= mh / Hd
        for s in surfs:
            C = s["C"].reshape(-1, 3)
            Pc = C @ Ti[:3, :3].T + Ti[:3, 3]
            zc = -Pc[:, 2]
            ray = cam - C
            dist = np.linalg.norm(ray, axis=1)
            front = (zc > 0.2) & (dist < cfg["max_range_m"]) & ((ray @ s["n"]) / np.maximum(dist, 1e-6) > cos_max)
            if not front.any():
                continue
            zs = np.where(front, zc, 1.0)
            u = Km[0, 0] * Pc[:, 0] / zs + Km[0, 2]
            vv = -Km[1, 1] * Pc[:, 1] / zs + Km[1, 2]
            ud = (u * Wd / mw).astype(int)
            vd = (vv * Hd / mh).astype(int)
            inside = front & (u >= 0) & (u < mw) & (vv >= 0) & (vv < mh) & s["valid"].reshape(-1)
            if not inside.any():
                continue
            idx = np.where(inside)[0]
            dd = D[np.clip(vd[idx], 0, Hd - 1), np.clip(ud[idx], 0, Wd - 1)]
            vis = np.isfinite(dd) & (dd > zc[idx] - cfg["occl_m"])
            idx = idx[vis]
            if not len(idx):
                continue
            seen = s["seen"].reshape(-1)
            seen[idx] += 1
            ui, vi = u[idx].astype(int), vv[idx].astype(int)
            for ci in range(len(names)):
                sc = cls_mask[ci, vi, ui]
                hit = sc > 0
                if hit.any():
                    s["pos"][ci].reshape(-1)[idx[hit]] += 1
                    s["score"][ci].reshape(-1)[idx[hit]] += sc[hit]
                    s["frames"][ci].append(int(i))

    regions, n_single = [], 0
    area_cell = cell * cell
    for s in surfs:
        seen = np.maximum(s["seen"], 1)
        for ci, name in enumerate(names):
            pos = s["pos"][ci]
            ratio = pos / seen
            core = (pos >= cfg["min_views"]) & (ratio >= cfg["min_ratio"])
            cand, nc = ndimage.label(pos >= 1, structure=np.ones((3, 3)))
            n_single += nc - len(np.unique(cand[core & (cand > 0)]))     # candidate patches with no agreed cell
            lab, n = ndimage.label(core, structure=np.ones((3, 3)))
            for k in range(1, n + 1):
                m = lab == k
                if m.sum() * area_cell < cfg["min_area_m2"].get(name, cfg["min_area_m2"]["default"]):
                    continue
                near = ndimage.binary_dilation(m, iterations=2)
                lo_m = m & (ratio >= cfg["ratio_lo"])
                hi_m = near & (pos >= cfg["min_views"]) & (ratio >= cfg["ratio_hi"])
                vs, us = np.nonzero(m)
                a = float(m.sum()) * area_cell
                conf = float(np.clip((s["score"][ci][m] / np.maximum(pos[m], 1)).mean() * ratio[m].mean(), 0, 1))
                regions.append({
                    "surface_id": s["id"], "class": name,
                    "area": Measurement(a, float(lo_m.sum()) * area_cell, max(a, float(hi_m.sum()) * area_cell),
                                        unit="m2", method="multiview-vote:v0 (uncalibrated)").to_json(),
                    "extent": {"u_min": round(float(us.min()) * cell, 3), "u_max": round(float(us.max() + 1) * cell, 3),
                               "v_min": round(float(vs.min()) * cell, 3), "v_max": round(float(vs.max() + 1) * cell, 3)},
                    "confidence": round(conf, 3),
                    "evidence": {"frames": [str(f) for f in sorted(set(s["frames"][ci]))[:8]],
                                 "rationale": f"{int(np.median(pos[m]))} of {int(np.median(s['seen'][m]))} views that saw "
                                              f"it agree (median over its cells)",
                                 "classifier": model["version"]},
                })
    regions.sort(key=lambda r: (r["surface_id"], r["class"], r["extent"]["u_min"]))
    for k, r in enumerate(regions):
        r["id"] = f"D{k + 1}"
    if n_single:
        warnings.append(f"damage: {n_single} candidate patches seen in too few views or inconsistently across views "
                        "(reflection, shadow, one-off detection): not reported")
    return regions, model
