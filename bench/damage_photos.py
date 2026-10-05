"""Damage detector on real photos (benchmark/raw/damage_photos, Wikimedia Commons, SOURCES.csv), image level.

    python bench/damage_photos.py [--out out/damage_photos]

Each photo is labelled with one class (or `negative`, a clean interior with confusers). The worker runs once with
class_threshold 0 (config/damage.yaml otherwise); the threshold is applied afterwards. Split: odd-numbered photos
of each class tune (class_threshold chosen by F1 over the tuning half), even-numbered photos report. A photo counts as:
- hit: a detection of its own class above threshold;  miss: none;
- false alarm: a detection of a class the photo does not show (negatives: any detection). Secondary damage named
  in the notes (e.g. staining next to a crack) is not counted against the detector.
This is per frame only. Region extent on a surface needs captures of damaged rooms, which the benchmark lacks.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PHOTOS = ROOT / "benchmark/raw/damage_photos"
SECONDARY = {"crack_05.jpg": {"water_stain"}, "water_stain_04.jpg": {"peeling_paint"},
             "peeling_paint_03.jpg": {"water_stain"}}
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def run(files: list[Path], out: Path) -> dict:
    import yaml
    cfg = yaml.safe_load((ROOT / "config/damage.yaml").read_text())
    job = {"images": [str(f) for f in files], "rot90": [0] * len(files), "classes": cfg["classes"],
           "describe": cfg["describe"], "negatives": cfg["negatives"], "class_threshold": 0.0,
           "box_threshold": cfg["box_threshold"], "text_threshold": cfg["text_threshold"],
           "work_px": cfg["work_px"], "max_box_frac": cfg["max_box_frac"]}
    key = abs(hash(json.dumps({k: v for k, v in job.items() if k != "images"}, sort_keys=True))) % 10**8
    op = out / f"dets_{key}.npz"
    if not op.exists():
        jp = out / "job.json"
        jp.write_text(json.dumps(job))
        subprocess.run([sys.executable, "-m", "roomscope.damage.worker", str(jp), str(op)], check=True,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    z = np.load(op)
    names = list(z["classes"])
    return {f.name: [(names[int(d[0])], float(d[1])) for d in z[f"det_{i}"]] for i, f in enumerate(files)}


def score(dets: dict, labels: dict, thr: float, files: list[str]) -> dict:
    hit = miss = fa = 0
    per = {}
    for f in files:
        got = {c for c, s in dets[f] if s >= thr}
        lab = labels[f]
        ok = lab != "negative" and lab in got
        wrong = got - {lab} - SECONDARY.get(f, set())
        hit += ok
        miss += lab != "negative" and not ok
        fa += bool(wrong)
        per[f] = {"label": lab, "detected": sorted(got)}
    pos = sum(labels[f] != "negative" for f in files)
    rec = hit / max(pos, 1)
    prec = hit / max(hit + fa, 1)
    return {"recall": round(rec, 3), "false_alarm_photos": fa, "n": len(files), "positives": pos,
            "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3), "per_photo": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "out/damage_photos")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(PHOTOS / "SOURCES.csv")))
    labels = {r["file"]: r["class"] for r in rows}
    files = sorted(PHOTOS / f for f in labels)
    tune = [f.name for f in files if int(f.stem.split("_")[-1]) % 2 == 1]
    test = [f.name for f in files if int(f.stem.split("_")[-1]) % 2 == 0]
    dets = run(files, a.out)
    grid = [(thr, score(dets, labels, thr, tune)) for thr in THRESHOLDS]
    for thr, s in grid:
        print(f"tune  class threshold {thr:.2f}: recall {s['recall']:.2f}  false-alarm photos {s['false_alarm_photos']}/{s['n']}  F1 {s['f1']:.2f}")
    thr, _ = max(grid, key=lambda g: (g[1]["f1"], g[0]))
    rep = score(dets, labels, thr, test)
    neg = [f for f in test if labels[f] == "negative"]
    print(f"chosen on the tuning half: class threshold {thr}")
    print(f"TEST  negatives with any detection: {sum(bool(rep['per_photo'][f]['detected']) for f in neg)}/{len(neg)}")
    print(f"TEST  recall {rep['recall']:.2f} ({round(rep['recall'] * rep['positives'])}/{rep['positives']})  "
          f"false-alarm photos {rep['false_alarm_photos']}/{rep['n']}  F1 {rep['f1']:.2f}")
    for cls in sorted({labels[f] for f in test}):
        fs = [f for f in test if labels[f] == cls]
        print(f"  {cls:14}", "  ".join(f"{f.split('_')[-1][:2]}:{','.join(rep['per_photo'][f]['detected']) or '-'}" for f in fs))
    (a.out / "report.json").write_text(json.dumps({"chosen": {"class_threshold": thr},
                                                   "tune": [{"thr": t, **{k: v for k, v in s.items() if k != "per_photo"}}
                                                            for t, s in grid],
                                                   "test": rep}, indent=1))


if __name__ == "__main__":
    main()
