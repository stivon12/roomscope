"""Fit conformal calibration for one tier from laser-scored captures, report leave-one-room-out
coverage, and write config/calibration.json (other tiers' entries are kept).

    python bench/calibrate.py --tier lidar [--level 0.9] [--rerun]

Runs the pipeline uncalibrated on every capture in bench/scenes.yaml that is on disk (cached under
out/calib/<tier>/ unless --rerun), scores it against the Faro laser, and collects one record per wall
length and per ceiling height.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from roomscope.core import calibrate as C
from roomscope.eval.laser import find_laser_dir, score_capture
from roomscope.pipeline import run_capture


def wall_bin(res: dict, wall_id: str) -> str:
    for rm in res["rooms"]:
        ids = [w["id"] for w in rm["walls"]]
        if wall_id in ids:
            return C.corner_bin(rm["walls"], ids.index(wall_id))
    return "supported"


def add_bins(recs: list[dict], tier: str) -> list[dict]:
    """Bins for records saved before bins existed (computed from the cached result.json)."""
    for r in recs:
        if r["kind"] == "wall_length" and "bin" not in r:
            res = json.loads((Path("out/calib") / tier / r["capture"] / "result.json").read_text())
            r["bin"] = wall_bin(res, r["id"])
    return recs


def collect(tier: str, rerun: bool) -> list[dict]:
    cfg = yaml.safe_load(Path("bench/scenes.yaml").read_text())
    root = Path(cfg["root"])
    recs = []
    for c in cfg["captures"]:
        scene = root / c["id"]
        if not scene.exists() or find_laser_dir(scene) is None:
            print(f"  skip {c['id']}: not downloaded")
            continue
        out = Path("out/calib") / tier
        res_p = out / c["id"] / "result.json"
        if rerun or not res_p.exists():
            try:
                if tier == "photo":
                    from make_photos import make
                    run_capture(make(scene), tier, out, calibrate=False)
                else:
                    run_capture(scene, tier, out, calibrate=False)
            except Exception as e:                       # a failed capture is reported, not hidden
                print(f"  FAIL {c['id']}: {type(e).__name__}: {e}")
                continue
        res = json.loads(res_p.read_text())
        sc = score_capture(out / c["id"], scene)
        walls = {w["id"]: w for rm in res["rooms"] for w in rm["walls"]}
        rooms = {rm["id"]: rm for rm in res["rooms"]}
        for w in sc["walls"]:
            m = walls[w["wall"]]["length"]
            recs.append({"tier": tier, "kind": "wall_length", "room": c["room"], "capture": c["id"], "id": w["wall"],
                         "value": m["value"], "truth": w["ref"], "raw_half": m["hi"] - m["value"],
                         "bin": wall_bin(res, w["wall"])})
        for h in sc["ceil"]:
            m = rooms[h["room"]]["ceiling_height"]
            recs.append({"tier": tier, "kind": "ceiling_height", "room": c["room"], "capture": c["id"], "id": h["room"],
                         "value": m["value"], "truth": h["ref"], "raw_half": m["hi"] - m["value"]})
        print(f"  {c['id']}: {len(sc['walls'])} walls, {len(sc['ceil'])} ceilings")
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["lidar", "video", "photo"])
    ap.add_argument("--level", type=float, default=0.9)
    ap.add_argument("--rerun", action="store_true")
    ap.add_argument("--from-records", action="store_true", help="refit from out/calib/records_<tier>.json")
    a = ap.parse_args()
    rec_p = Path(f"out/calib/records_{a.tier}.json")
    if a.from_records:
        recs = add_bins(json.loads(rec_p.read_text()), a.tier)
    else:
        recs = collect(a.tier, a.rerun)
        Path("out/calib").mkdir(parents=True, exist_ok=True)
        rec_p.write_text(json.dumps(recs, indent=1))
    fitted = C.fit(recs, a.level)
    cfg = C.load()
    cfg[a.tier] = fitted[a.tier]
    C.CONFIG.write_text(json.dumps(cfg, indent=2))
    for kind, v in fitted[a.tier].items():
        cov = v["loro_coverage"]
        ci = v["loro_coverage_ci95"]
        print(f"{a.tier:6s} {kind:15s} n={v['n']:3d} rooms={v['n_rooms']}  q={v['q']:.2f} ({v['method']})")
        print(f"    raw coverage {v['raw_coverage']:.2f} -> leave-one-room-out "
              + (f"{cov:.2f} [95% CI {ci[0]:.2f}-{ci[1]:.2f}]" if cov is not None else "n/a")
              + f" (target {v['level']:.2f}, nominal {v['nominal_level']:.2f}; {v['loro_infinite_folds']} folds too small)")
        print(f"    median half-width {(v['loro_median_halfwidth'] or float('nan')) * 100:.1f} cm, "
              f"median |err| {v['abs_err_median'] * 100:.1f} cm, width/error {v['width_to_error'] or float('nan'):.1f}, "
              f"interval score {(v['loro_interval_score'] or float('nan')) * 100:.1f} cm; per room {v['loro_per_room']}")


if __name__ == "__main__":
    main()
