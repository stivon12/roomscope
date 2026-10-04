"""Split-conformal calibration of every reported interval, per tier and per quantity.

Why conformal: the raw intervals come from plane-fit statistics, i.e. millimetres from thousands of
points, while the real errors are systematic centimetres: which surface was taken as the wall,
residual depth bias, pose drift. A model-based error budget would have to enumerate those causes.
Split conformal needs only (prediction, truth) pairs from held-out captures and gives finite-sample
marginal coverage under exchangeability.

Score (normalised, so intervals stay adaptive):
    s = |value - truth| / u,   u = sqrt(sigma_raw^2 + a^2 + (b * value)^2)
- sigma_raw is the pipeline's own standard error. It is larger for walls observed over less of their
  length, so those get wider intervals.
- a = 1 cm and b = 0.5% are a floor: without it a 1 mm plane-fit error would claim mm intervals.
Calibrated half-width = q * u, where q is the ceil((n+1)(1-alpha))-th smallest calibration score.

Coverage is reported by leave-one-ROOM-out: captures of the same room are not exchangeable with each
other, so a room's own repeat captures must not calibrate it.

Quantities with ground truth: wall length, ceiling height. Areas inherit the wall-length multiplier
through their raw error propagation. Opening dimensions have no ground truth in the laser data yet;
their intervals carry the wall-length multiplier and are labelled "conformal-transfer".
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parents[3] / "config" / "calibration.json"
Z_RAW = 1.645          # raw intervals are value +- Z_RAW * sigma_raw (layout.meas_len)
A_FLOOR, B_FLOOR = 0.01, 0.005


def normaliser(value: float, raw_half: float) -> float:
    s = raw_half / Z_RAW
    return math.sqrt(s * s + A_FLOOR ** 2 + (B_FLOOR * abs(value)) ** 2)


def conformal_q(scores: np.ndarray, level: float) -> float:
    n = len(scores)
    k = math.ceil((n + 1) * level)
    if n == 0 or k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def fit(records: list[dict], level: float = 0.9) -> dict:
    """records: {tier, kind, room, value, truth, raw_half}. Returns {tier: {kind: {...}}}."""
    out: dict = {}
    for tier in sorted({r["tier"] for r in records}):
        out[tier] = {}
        for kind in sorted({r["kind"] for r in records if r["tier"] == tier}):
            R = [r for r in records if r["tier"] == tier and r["kind"] == kind]
            s = np.array([abs(r["value"] - r["truth"]) / normaliser(r["value"], r["raw_half"]) for r in R])
            rooms = np.array([r["room"] for r in R])
            # leave-one-room-out coverage and width
            cov, wid, inf_folds = [], [], 0
            for rm in np.unique(rooms):
                q = conformal_q(s[rooms != rm], level)
                if not math.isfinite(q):          # too few held-in points for a finite interval
                    inf_folds += 1
                    continue
                for r, si in zip(np.array(R, dtype=object)[rooms == rm], s[rooms == rm]):
                    cov.append(si <= q)
                    wid.append(q * normaliser(r["value"], r["raw_half"]) if math.isfinite(q) else float("inf"))
            raw_cov = np.mean([abs(r["value"] - r["truth"]) <= r["raw_half"] for r in R])
            out[tier][kind] = {
                "q": conformal_q(s, level), "level": level, "n": len(R), "n_rooms": int(len(np.unique(rooms))),
                "loro_coverage": float(np.mean(cov)) if cov else None,
                "loro_median_halfwidth": float(np.median(wid)) if wid else None,
                "loro_infinite_folds": inf_folds,
                "raw_coverage": float(raw_cov),
                "abs_err_median": float(np.median([abs(r["value"] - r["truth"]) for r in R])),
            }
    return out


def load() -> dict:
    return json.loads(CONFIG.read_text()) if CONFIG.exists() else {}


def _recal(m: dict, q: float, tag: str, level: float, ref_scale: float | None = None) -> dict:
    """Rewrite one measurement dict in place: half-width = q * u (or raw half-width * ref_scale)."""
    v, raw_half = m["value"], m["hi"] - m["value"]
    half = raw_half * ref_scale if ref_scale is not None else q * normaliser(v, raw_half)
    m.update(lo=round(v - half, 4), hi=round(v + half, 4), level=level, method=tag)
    return m


def apply(result: dict, tier: str, cfg: dict | None = None) -> dict:
    """Calibrate every measurement in a result dict for `tier`. Leaves raw intervals (and says so) when
    no calibration exists for that tier."""
    cfg = load() if cfg is None else cfg
    c = cfg.get(tier, {})
    if not c:
        result.setdefault("warnings", []).append(f"no calibration for tier '{tier}': intervals are raw fit statistics")
        return result
    L, H = c.get("wall_length"), c.get("ceiling_height")
    for room in result["rooms"]:
        if H and math.isfinite(H["q"]):
            _recal(room["ceiling_height"], H["q"], f"conformal:{tier}:ceiling_height:v1", H["level"])
            for w in room["walls"]:
                w["height"] = dict(room["ceiling_height"])
        if L and math.isfinite(L["q"]):
            ratios = []
            for w in room["walls"]:
                raw = w["length"]["hi"] - w["length"]["value"]
                _recal(w["length"], L["q"], f"conformal:{tier}:wall_length:v1", L["level"])
                ratios.append((w["length"]["hi"] - w["length"]["value"]) / max(raw, 1e-9))
            k = float(np.median(ratios)) if ratios else 1.0
            for key in ("floor_area", "perimeter"):
                if key in room:
                    _recal(room[key], 0, f"conformal-propagated:{tier}:wall_length:v1", L["level"], ref_scale=k)
            for o in room["openings"]:
                for key in ("width", "height", "sill", "offset"):
                    if key in o:
                        _recal(o[key], L["q"], f"conformal-transfer:{tier}:wall_length:v1 (no opening GT yet)",
                               L["level"])
    if L and math.isfinite(L["q"]):
        for s in result.get("surfaces", []):
            if s["kind"] == "wall":
                w = next((w for rm in result["rooms"] for w in rm["walls"] if w["id"] == s.get("wall_id")), None)
                if w:
                    rel = math.hypot((w["length"]["hi"] - w["length"]["value"]) / max(w["length"]["value"], 1e-6),
                                     (w["height"]["hi"] - w["height"]["value"]) / max(w["height"]["value"], 1e-6))
                    v = s["area"]["value"]
                    s["area"].update(lo=round(v * (1 - rel), 4), hi=round(v * (1 + rel), 4), level=L["level"],
                                     method=f"conformal-propagated:{tier}:v1")
            else:
                fa = next(rm["floor_area"] for rm in result["rooms"] if rm["id"] == s["room_id"])
                s["area"] = dict(fa)
        fp = result["footprint"]["area"]
        halves = [rm["floor_area"]["hi"] - rm["floor_area"]["value"] for rm in result["rooms"]]
        h = math.sqrt(sum(x * x for x in halves))
        fp.update(lo=round(fp["value"] - h, 4), hi=round(fp["value"] + h, 4), level=L["level"],
                  method=f"conformal-propagated:{tier}:v1")
    return result
