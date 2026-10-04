"""Audit the large wall-length errors (docs/RESEARCH.md section 3; checklist from the corner research).

A wall's length error is set by its two neighbours: length = distance between the laser reference
positions of the walls before and after it. For every scored wall with |error| >= --min-err this prints,
for each neighbour: its plane offset error, whether its edge came from the floor-boundary fallback, and
every laser surface within 25 cm of it (offset from our line, point count). Two laser surfaces near a
neighbour means the reference itself is ambiguous (scorer risk); one surface far from our line means our
plane is on the wrong layer; a fallback edge means no wall plane was fitted there.

    python bench/wall_audit.py [--tier lidar] [--min-err 0.09]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from diag_lidar import wall_surfaces
from roomscope.eval.laser import score_capture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="lidar")
    ap.add_argument("--min-err", type=float, default=0.09)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path("bench/scenes.yaml").read_text())
    root = Path(cfg["root"])
    rows = []
    for c in cfg["captures"]:
        out = Path("out/calib") / a.tier / c["id"]
        if not (out / "result.json").exists():
            continue
        res = json.loads((out / "result.json").read_text())
        sc, ref = score_capture(out, root / c["id"], return_ref=True)
        surf = {r["wall"]: r for r in wall_surfaces(res, ref)}
        perr = {r["wall"]: r["offset_err"] for r in sc["wall_planes"]}
        fallback = {w.split(":")[0] for w in res.get("warnings", []) if "no fitted wall plane" in w}
        for rm in res["rooms"]:
            ids = [w["id"] for w in rm["walls"]]
            for w in sc["walls"]:
                if w["room"] != rm["id"] or abs(w["err"]) < a.min_err:
                    continue
                k = ids.index(w["wall"])
                nb = [ids[k - 1], ids[(k + 1) % len(ids)]]
                row = {"capture": c["id"], "wall": w["wall"], "len": round(w["ref"] + w["err"], 2),
                       "err_cm": round(w["err"] * 100, 1),
                       "neighbours": [{"wall": n, "offset_err_cm": round(perr.get(n, float("nan")) * 100, 1),
                                       "fallback": n in fallback,
                                       "laser_surfaces_cm": surf.get(n, {}).get("surfaces")} for n in nb]}
                rows.append(row)
                print(json.dumps(row), flush=True)
    Path("out/diag").mkdir(parents=True, exist_ok=True)
    Path(f"out/diag/wall_audit_{a.tier}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
