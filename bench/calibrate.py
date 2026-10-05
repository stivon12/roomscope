"""Fit conformal calibration for one tier from laser-scored captures, report leave-one-room-out
coverage, and write config/calibration.json (other tiers' entries are kept).

    python bench/calibrate.py --tier lidar [--level 0.9] [--rerun]

Runs the pipeline uncalibrated on every calibration capture on disk (bench/scenes.yaml ARKitScenes + MuSHRoom,
minus every room of the benchmark set in bench/set.yaml; cached under out/calib/<tier>/ unless --rerun), scores
it against the Faro laser (structural reference, clean walls only), and collects one record per wall length and
per ceiling height.
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
            res = json.loads(next((Path("out/calib") / tier / r["capture"]).glob("*/result.json")).read_text())
            r["bin"] = wall_bin(res, r["id"])
    return recs


MUSHROOM = {"vr_room": ["vr_room_long", "vr_room_short"], "coffee_room": ["coffee_room_long", "coffee_room_short"],
            "honka": ["honka_long", "honka_short"]}


def calibration_captures() -> list[dict]:
    """Laser-scored captures for calibration: bench/scenes.yaml (ARKitScenes) + MuSHRoom, minus every ROOM that is
    in the benchmark set (bench/set.yaml groups), so the gates are never measured on a room that helped calibrate."""
    held = {e["group"] for e in yaml.safe_load(Path("bench/set.yaml").read_text())["captures"]}
    cfg = yaml.safe_load(Path("bench/scenes.yaml").read_text())
    caps = [{"id": c["id"], "room": c["room"], "source": "arkitscenes", "path": Path(cfg["root"]) / c["id"]}
            for c in cfg["captures"]]
    caps += [{"id": cid, "room": room, "source": "mushroom", "path": Path("benchmark/raw/mushroom_stray") / cid,
              "ply": Path(f"benchmark/raw/mushroom/room_datasets/{room}/gt_pd.ply")}
             for room, ids in MUSHROOM.items() for cid in ids]
    return [c for c in caps if c["room"] not in held]


def tier_input(c: dict, tier: str) -> Path:
    if tier == "lidar":
        return c["path"]
    if tier == "video":
        return c["path"] / "rgb.mp4" if c["source"] == "mushroom" else c["path"]
    from make_photos import make, make_stray
    if c["source"] == "mushroom":
        d = Path("out/photo_inputs") / c["id"]
        return d if d.exists() else make_stray(c["path"])
    return make(c["path"])


def collect(tier: str, rerun: bool) -> list[dict]:
    recs = []
    for c in calibration_captures():
        if not c["path"].exists() or (c["source"] == "arkitscenes" and find_laser_dir(c["path"]) is None):
            print(f"  skip {c['id']}: not downloaded")
            continue
        out = Path("out/calib") / tier / c["id"]
        hits = list(out.glob("*/result.json"))
        if rerun or not hits:
            try:
                run_capture(tier_input(c, tier), tier, out, calibrate=False, damage=False)
            except Exception as e:                       # a failed capture is reported, not hidden
                print(f"  FAIL {c['id']}: {type(e).__name__}: {e}")
                continue
            hits = list(out.glob("*/result.json"))
        res_dir = hits[0].parent
        res = json.loads((res_dir / "result.json").read_text())
        if c["source"] == "mushroom":
            from mushroom_eval import score
            sc = score(res_dir, c["ply"])
        else:
            sc = score_capture(res_dir, c["path"])
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
        print(f"  {c['id']}: {len(sc['walls'])} walls, {len(sc['ceil'])} ceilings", flush=True)
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
