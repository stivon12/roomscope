"""Room-partition accuracy on Bormann et al.'s labelled maps (ICRA 2016), to set the SCP threshold band
BEFORE it is used on our captures (core/rooms_scp.py, docs/ROOM_SEGMENTATION.md).

    python bench/room_seg_eval.py [--maps third_party/ipa_coverage_planning/.../test_maps] [--sweep]

Maps: 0.05 m/px, 255 free; ground truth = connected free regions of <map>_gt_segmentation.png (rooms are
closed off with drawn lines), regions > 1 m2. Each map is scored without furniture and, when the
<map>_furnitures.png variant exists, with it. Metrics:
- Bormann recall / precision: mean over GT rooms of max overlap / GT area; mean over segments of max
  overlap / segment area (their Table II: Voronoi 86.6 / 94.5 with furniture);
- room F1 at IoU > 0.5 (one-to-one greedy matching), and counts of over-segmented GT rooms (>= 2
  segments each covering >= 20 % of it) and under-segmenting segments (covering >= 50 % of >= 2 GT rooms).
The maps are LGPL / non-commercial (Fraunhofer IPA): used here only to evaluate, not shipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from roomscope.core import rooms_scp as S

MAPS = Path("third_party/ipa_coverage_planning/ipa_room_segmentation/common/files/test_maps")
CELL = 0.05


def gt_rooms(path: Path) -> np.ndarray:
    g = (cv2.imread(str(path), 0) > 250).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(g, connectivity=4)
    keep = np.flatnonzero(st[:, 4] * CELL * CELL > 1.0)
    keep = keep[keep > 0]
    out = np.zeros_like(lab)
    for k, v in enumerate(keep, 1):
        out[lab == v] = k
    return out


def score(pred: np.ndarray, gt: np.ndarray) -> dict:
    G, P = int(gt.max()), int(pred.max())
    m = (gt > 0) & (pred > 0)
    C = np.zeros((G + 1, P + 1), np.int64)
    np.add.at(C, (gt[m], pred[m]), 1)
    ga = np.bincount(gt.ravel(), minlength=G + 1)
    pa = np.bincount(pred.ravel(), minlength=P + 1)
    rec = float(np.mean([C[g].max() / ga[g] for g in range(1, G + 1)])) if G else 0.0
    prec = float(np.mean([C[:, p].max() / pa[p] for p in range(1, P + 1)])) if P else 0.0
    iou = C / np.maximum(ga[:, None] + pa[None, :] - C, 1)
    pairs, used_g, used_p = [], set(), set()
    for g, p in sorted(((g, p) for g in range(1, G + 1) for p in range(1, P + 1) if iou[g, p] > 0.5),
                       key=lambda t: -iou[t]):
        if g not in used_g and p not in used_p:
            pairs.append((g, p)); used_g.add(g); used_p.add(p)
    tp = len(pairs)
    f1 = 2 * tp / max(G + P, 1)
    over = sum(1 for g in range(1, G + 1) if (C[g, 1:] >= 0.2 * ga[g]).sum() >= 2)
    under = sum(1 for p in range(1, P + 1) if (C[1:, p] >= 0.5 * ga[1:]).sum() >= 2)
    return {"recall": rec, "precision": prec, "f1": f1, "gt": G, "pred": P, "over": over, "under": under}


def run(cfgs: list[dict], maps: Path) -> dict:
    names = sorted(p.name.replace("_gt_segmentation.png", "") for p in maps.glob("*_gt_segmentation.png"))
    res = {i: [] for i in range(len(cfgs))}
    for n in names:
        gt = gt_rooms(maps / f"{n}_gt_segmentation.png")
        for variant in ("", "_furnitures"):
            mp = maps / f"{n}{variant}.png"
            if not mp.exists():
                continue
            free = cv2.imread(str(mp), 0) > 250
            D = S.clearance(free, CELL)
            for i, c in enumerate(cfgs):
                L = np.log(np.maximum(D, c["r_floor"]))
                if i == 0 or c["r_floor"] != cfgs[i - 1]["r_floor"]:
                    peaks = S.persistence_peaks(L, free, np.log(c["r_floor"]))
                tau = c["tau"] if "tau" in c else S.choose_tau(peaks, tuple(c["band"]), c["r_min"])
                seeds = [pk for pk in peaks if pk.r_birth >= c["r_min"] and pk.persistence > tau] or \
                    [max(peaks, key=lambda pk: pk.r_birth)]
                from skimage.segmentation import watershed
                mk = np.zeros(free.shape, np.int32)
                for k, pk in enumerate(seeds, 1):
                    mk[pk.ij] = k
                lab = watershed(-L, mk, mask=free)
                s = score(lab, gt)
                s.update(map=n, furniture=bool(variant), tau=tau)
                res[i].append(s)
    return {"configs": cfgs, "results": res}


def summary(rows: list[dict]) -> dict:
    out = {}
    for furn in (False, True):
        r = [x for x in rows if x["furniture"] == furn]
        if r:
            out["furnished" if furn else "plain"] = {
                "maps": len(r), "recall": round(100 * np.mean([x["recall"] for x in r]), 1),
                "precision": round(100 * np.mean([x["precision"] for x in r]), 1),
                "room_f1": round(np.mean([x["f1"] for x in r]), 3),
                "over": int(sum(x["over"] for x in r)), "under": int(sum(x["under"] for x in r)),
                "gt_rooms": int(sum(x["gt"] for x in r)), "pred_rooms": int(sum(x["pred"] for x in r))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", type=Path, default=MAPS)
    ap.add_argument("--sweep", action="store_true", help="sweep fixed tau and band choices")
    ap.add_argument("--out", type=Path, default=Path("out/room_seg_eval.json"))
    a = ap.parse_args()
    if a.sweep:
        cfgs = [{"tau": float(np.log(t)), "r_min": 0.6, "r_floor": 0.05} for t in (1.2, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0)]
        cfgs += [{"band": [float(np.log(lo)), float(np.log(hi))], "r_min": 0.6, "r_floor": 0.05}
                 for lo, hi in ((1.3, 2.0), (1.5, 2.5), (1.4, 3.0))]
    else:
        import yaml
        c = yaml.safe_load(Path("config/layout.yaml").read_text())["rooms"]["scp"]
        cfgs = [{"band": [float(np.log(c["ratio_band"][0])), float(np.log(c["ratio_band"][1]))],
                 "r_min": c["r_min_m"], "r_floor": c["r_floor_m"]}]
    r = run(cfgs, a.maps)
    table = []
    for i, c in enumerate(cfgs):
        lab = (f"tau=ratio {np.exp(c['tau']):.2f}" if "tau" in c
               else f"gap in ratio band [{np.exp(c['band'][0]):.2f}, {np.exp(c['band'][1]):.2f}]")
        table.append({"config": lab, **summary(r["results"][i])})
        print(json.dumps(table[-1]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": table, "per_map": {str(i): v for i, v in r["results"].items()}}, indent=1))


if __name__ == "__main__":
    main()
