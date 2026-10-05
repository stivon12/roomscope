"""Every gate of the brief, measured on the benchmark set (bench/set.yaml), written to bench/results/gates.{json,md}.

    python bench/gates.py run   [--tiers lidar,video,photo] [--only id,...]   # pipeline runs, cached in out/bench
    python bench/gates.py score                                              # score cached runs -> gates

Runs are the shipped pipeline (calibrated), without damage detection (not a round-1 gate; its own check is
bench/damage_photos.py). One tier/capture at a time (memory). Scoring:
- laser captures: eval/laser.score_capture (ARKitScenes) or bench/mushroom_eval.score (MuSHRoom Faro .ply), with
  the structural wall reference: only `wall`-status walls are scored, `step`/unscored are counted, not hidden;
- repeatability: bench/repeatability.compare on every pair of captures of one group at one tier (no GT needed).
A gate that the set cannot measure is listed as such with the reason, never as passed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
OUT = ROOT / "out" / "bench"
RES = ROOT / "bench" / "results"


def entries() -> list[dict]:
    return yaml.safe_load((ROOT / "bench" / "set.yaml").read_text())["captures"]


def tiers_of(e: dict) -> list[str]:
    return e.get("tiers", ["lidar", "video", "photo"])


def input_for(e: dict, tier: str) -> Path:
    p = ROOT / e["path"]
    if tier == "lidar":
        return p
    if tier == "video":
        return p / "rgb.mp4" if (p / "rgb.mp4").exists() else p
    import make_photos as MP                       # photo tier input, built once
    if e["source"] == "arkitscenes":
        return MP.make(p)
    sel = ROOT / e["photo_select"] if e.get("photo_select") else None
    d = ROOT / "out" / "photo_inputs" / (p.name + ("_rooms" if sel else ""))
    return d if d.exists() else MP.make_stray(p, select=sel)


def result_dir(e: dict, tier: str) -> Path:
    base = OUT / tier / e["id"]
    inp = input_for(e, tier) if tier != "photo" else None
    name = inp.name if inp is not None else None
    if tier == "photo":
        hits = list(base.glob("*/result.json"))
        return hits[0].parent if hits else base / "_missing"
    return base / name


def run(tiers: list[str], only: list[str] | None):
    from roomscope.pipeline import run_capture
    for tier in tiers:
        for e in entries():
            if tier not in tiers_of(e) or (only and e["id"] not in only):
                continue
            if (result_dir(e, tier) / "result.json").exists():
                continue
            print(f"[gates] {tier} {e['id']}", flush=True)
            try:
                run_capture(input_for(e, tier), tier, OUT / tier / e["id"], damage=False)   # not a round-1 gate
            except Exception as ex:                                    # reported in the gates, not hidden
                (OUT / tier / e["id"]).mkdir(parents=True, exist_ok=True)
                (OUT / tier / e["id"] / "FAILED.txt").write_text(f"{type(ex).__name__}: {ex}")
                print(f"  FAILED: {type(ex).__name__}: {ex}", flush=True)


def score_one(e: dict, d: Path) -> dict | None:
    if e["gt"] == "none":
        return None
    if e["gt"] == "faro_arkitscenes":
        from roomscope.eval.laser import score_capture
        return score_capture(d, ROOT / e["path"])
    import mushroom_eval
    return mushroom_eval.score(d, ROOT / e["gt"].split(":", 1)[1])


def _rate(xs) -> dict:
    xs = list(xs)
    return {"n": len(xs), "pass": int(sum(xs)), "rate": round(float(np.mean(xs)), 3) if xs else None}


def score():
    from repeatability import compare
    per: dict = {}
    for e in entries():
        for tier in tiers_of(e):
            d = result_dir(e, tier)
            rec = {"id": e["id"], "tier": tier, "group": e["group"]}
            if not (d / "result.json").exists():
                fail = OUT / tier / e["id"] / "FAILED.txt"
                rec["status"] = fail.read_text() if fail.exists() else "not run"
                per[(e["id"], tier)] = rec
                continue
            res = json.loads((d / "result.json").read_text())
            rec.update(status="ok", rooms=len(res["rooms"]), dir=str(d.relative_to(ROOT)),
                       stitch=res["footprint"]["stitch_status"])
            try:
                sc = score_one(e, d)
            except Exception as ex:
                rec["score_error"] = f"{type(ex).__name__}: {ex}"
                sc = None
            if sc is not None:
                rec.update(ceil=sc["ceil"], walls=sc["walls"], walls_step=sc["walls_step"],
                           planes=sc["wall_planes"], unscored=len(sc["unscored"]),
                           n_edges=len(sc["wall_planes"]) + len(sc["wall_planes_step"]) + len(sc["unscored"]))
            per[(e["id"], tier)] = rec
    recs = list(per.values())
    ok = [r for r in recs if r["status"] == "ok"]

    def walls(tier):
        return [w for r in ok if r["tier"] == tier for w in r.get("walls", [])]

    def ceils(tier):
        return [c for r in ok if r["tier"] == tier for c in r.get("ceil", [])]

    gates = []
    # G.2 ceiling height (LiDAR) <= 1.5 cm per room vs laser
    c = ceils("lidar")
    gates.append({"id": "G.2a", "gate": "Ceiling height <= 1.5 cm (LiDAR, per room vs laser)", "target": 1.0,
                  **_rate(abs(x["err"]) <= 0.015 for x in c),
                  "detail": f"|err| median {100 * np.median([abs(x['err']) for x in c]):.2f} cm, max "
                            f"{100 * max(abs(x['err']) for x in c):.2f} cm" if c else ""})
    # G.2 multi-capture spread <= 1 cm (same room, LiDAR)
    spreads = []
    for g, rs in itertools.groupby(sorted(ok, key=lambda r: r["group"]), key=lambda r: r["group"]):
        hs = [x["ref"] + x["err"] for r in rs if r["tier"] == "lidar" for x in r.get("ceil", [])[:1]]
        if len(hs) >= 2:
            spreads.append((g, max(hs) - min(hs)))
    gates.append({"id": "G.2b", "gate": "Ceiling spread across captures of one room <= 1 cm (LiDAR)", "target": 1.0,
                  **_rate(s <= 0.01 for _, s in spreads),
                  "detail": ", ".join(f"{g} {100 * s:.2f} cm" for g, s in spreads)})
    # G.3 repeatability: matched walls of repeat captures agree within max(1 cm, 0.5 %)
    rep = []
    rep_detail = []
    for tier in ("lidar", "video", "photo"):
        by = {}
        for r in ok:
            if r["tier"] == tier:
                by.setdefault(r["group"], []).append(r)
        for g, rs in by.items():
            for a, b in itertools.combinations(rs, 2):
                try:
                    cmp = compare(ROOT / a["dir"], ROOT / b["dir"])
                except Exception as ex:
                    rep_detail.append(f"{tier} {a['id']}/{b['id']}: {type(ex).__name__}")
                    continue
                ws = [w for rm in cmp["matched_rooms"] for w in rm["walls"]]
                for w in ws:
                    rep.append((tier, abs(w["diff"]) <= max(0.01, 0.005 * max(w["a"], w["b"]))))
                rep_detail.append(f"{tier} {a['id']}/{b['id']}: {len(ws)} walls matched, rooms unmatched "
                                  f"{len(cmp['unmatched_a'])}+{len(cmp['unmatched_b'])}")
    for tier in ("lidar", "video", "photo"):
        xs = [p for t, p in rep if t == tier]
        gates.append({"id": f"G.3-{tier}", "gate": f"Repeatability: wall lengths of two captures within max(1 cm, 0.5 %) ({tier})",
                      "target": 1.0, **_rate(xs), "detail": "; ".join(d for d in rep_detail if d.startswith(tier))})
    # G.6 wall length tolerance per tier (structural laser reference, clean walls only)
    for tier, tol in (("photo", 0.08), ("video", 0.03), ("lidar", 0.03)):
        w = walls(tier)
        gates.append({"id": f"G.6-{tier}", "gate": f"Wall length within +-{int(tol * 100)} % ({tier}, clean laser reference)"
                      + (" [not a brief gate for LiDAR: shown for comparison]" if tier == "lidar" else ""),
                      "target": 1.0, **_rate(abs(x["rel"]) <= tol for x in w),
                      "detail": f"median |rel| {100 * np.median([abs(x['rel']) for x in w]):.1f} %" if w else "no scored walls"})
    # calibration: intervals cover the truth at the nominal 0.9
    for tier in ("lidar", "video", "photo"):
        cov = [x["covered"] for x in walls(tier) + ceils(tier)]
        gates.append({"id": f"CAL-{tier}", "gate": f"Calibrated intervals cover the laser value (nominal 0.9) ({tier})",
                      "target": 0.9, **_rate(cov), "detail": "walls + ceilings"})
    # photo stitch of the multi-room set
    ms = [r for r in ok if r["tier"] == "photo" and r["id"] == "c7d28f72c6"]
    gates.append({"id": "G.5", "gate": "Photo tier: whole property stitched from per-room folders, footprint +-8 %",
                  "target": None, "n": len(ms), "pass": None, "rate": None,
                  "detail": (f"stitch_status={ms[0]['stitch']}, {ms[0]['rooms']} of 6 rooms placed; footprint accuracy "
                             "NOT measurable: the multi-room capture has no ground truth") if ms else "not run"})
    gates.append({"id": "G.1", "gate": "Openings <= 2 cm on >= 85 %", "target": 0.85, "n": 0, "pass": None, "rate": None,
                  "detail": "NOT measurable: no door/window ground truth in any laser scan of the set (needs annotation)"})
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "gates.json").write_text(json.dumps({"gates": gates, "captures": recs}, indent=1, default=float))
    lines = ["# Gates on the benchmark set", "", "Generated by `python bench/gates.py score` from runs in out/bench "
             "(bench/set.yaml; composition vs the brief in benchmark/SET.md).", "",
             "| gate | measured | target | detail |", "|---|---|---|---|"]
    for g in gates:
        m = "not measurable" if g["rate"] is None and g["n"] in (0, None) else (
            f"{g['pass']}/{g['n']} = {g['rate']:.2f}" if g["rate"] is not None else "see detail")
        lines.append(f"| {g['id']} {g['gate']} | {m} | {g['target'] if g['target'] is not None else '-'} | {g['detail']} |")
    lines += ["", "## Runs", "", "| capture | tier | status | rooms | clean walls | step walls | unscored edges |",
              "|---|---|---|---|---|---|---|"]
    for r in recs:
        lines.append(f"| {r['id']} | {r['tier']} | {r['status'][:60]} | {r.get('rooms', '')} | {len(r.get('walls', []))} | "
                     f"{len(r.get('walls_step', []))} | {r.get('unscored', '')} |")
    (RES / "gates.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:len(gates) + 6]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "score"])
    ap.add_argument("--tiers", default="lidar,video,photo")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if a.stage == "run":
        run(a.tiers.split(","), a.only.split(",") if a.only else None)
    else:
        score()
