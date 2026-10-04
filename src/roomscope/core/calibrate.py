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
other, so a room's own repeat captures must not calibrate it. The quantile itself is room-level
(pooled_q): each room has equal weight and the finite-sample correction counts rooms, not walls.
Every coverage figure carries a Clopper-Pearson 95% interval; with 5-7 rooms it is wide, and that is
the honest statement of what the benchmark can show.

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


def pooled_q(scores: np.ndarray, rooms: np.ndarray, level: float) -> float:
    """Room-weighted conformal quantile (CDF pooling, after Dunn, Wasserman & Ramdas, two-layer
    hierarchical CP): every room gets equal total weight, so a room with many walls cannot dominate.

    The finite-sample correction counts measurements (n), not rooms: a correction over rooms K needs
    K >= 9 for any finite 90% interval, and the benchmark has 4-7 rooms. So the guarantee assumes
    measurements are exchangeable across rooms; that assumption is checked, not assumed, by the
    leave-one-room-out coverage and its Clopper-Pearson interval reported next to every fit."""
    u = np.unique(rooms)
    K, n = len(u), len(scores)
    if K == 0:
        return float("inf")
    lvl = level * (1 + 1 / n)
    if lvl > 1:
        return float("inf")
    w = np.array([1.0 / (K * (rooms == r).sum()) for r in rooms])
    o = np.argsort(scores)
    cw = np.cumsum(w[o])
    return float(scores[o][min(np.searchsorted(cw, lvl - 1e-12), len(o) - 1)])


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    from scipy.stats import beta
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def interval_score(lo: float, hi: float, y: float, alpha: float) -> float:
    """Winkler / interval score (Gneiting & Raftery 2007): width + 2/alpha * miss distance."""
    return (hi - lo) + 2 / alpha * max(lo - y, 0) + 2 / alpha * max(y - hi, 0)


# training-conditional (PAC) margin: with ~5-7 calibration rooms one unlucky draw can undercover, and
# overconfidence caps the score, so we calibrate at a slightly higher nominal level (Vovk 2012)
PAC_MARGIN = 0.03


def _scores(R: list[dict]) -> np.ndarray:
    return np.array([abs(r["value"] - r["truth"]) / normaliser(r["value"], r["raw_half"]) for r in R])


def _loro(R: list[dict], s: np.ndarray, rooms: np.ndarray, level: float, pool=None) -> dict:
    """Leave-one-room-out: q from the other rooms (plus `pool` scores/rooms if given), tested on the left-out room."""
    cov, wid, isc, inf_folds, per_room = [], [], [], 0, {}
    alpha = 1 - level
    for rm in np.unique(rooms):
        tr_s, tr_r = s[rooms != rm], rooms[rooms != rm]
        if pool is not None:
            tr_s = np.r_[tr_s, pool[0][pool[1] != rm]]
            tr_r = np.r_[tr_r, pool[1][pool[1] != rm]]
        q = pooled_q(tr_s, tr_r, level)
        if not math.isfinite(q):
            inf_folds += 1
            continue
        c = []
        for r, si in zip(np.array(R, dtype=object)[rooms == rm], s[rooms == rm]):
            h = q * normaliser(r["value"], r["raw_half"])
            c.append(si <= q)
            wid.append(h)
            isc.append(interval_score(r["value"] - h, r["value"] + h, r["truth"], alpha))
        cov += c
        per_room[str(rm)] = round(float(np.mean(c)), 3)
    k, n = int(np.sum(cov)), len(cov)
    return {"loro_coverage": (k / n) if n else None,
            "loro_coverage_ci95": clopper_pearson(k, n) if n else None,
            "loro_median_halfwidth": float(np.median(wid)) if wid else None,
            "loro_interval_score": float(np.mean(isc)) if isc else None,
            "loro_infinite_folds": inf_folds, "loro_per_room": per_room}


def fit(records: list[dict], level: float = 0.9) -> dict:
    """records: {tier, kind, room, value, truth, raw_half}. Returns {tier: {kind: {...}}}.

    Quantities with too few rooms for a finite room-level quantile (ceiling heights: one per room) are
    pooled with the tier's wall lengths through the shared normalised score; this is used only if
    leave-one-room-out coverage of that quantity still reaches the target, and is labelled."""
    out: dict = {}
    nominal = min(level + PAC_MARGIN, 0.99)
    for tier in sorted({r["tier"] for r in records}):
        out[tier] = {}
        TR = [r for r in records if r["tier"] == tier]
        W = [r for r in TR if r["kind"] == "wall_length"]
        sW, rW = _scores(W), np.array([r["room"] for r in W])
        for kind in sorted({r["kind"] for r in TR}):
            R = [r for r in TR if r["kind"] == kind]
            s, rooms = _scores(R), np.array([r["room"] for r in R])
            # fallbacks, each labelled: drop the PAC margin; then the highest level n allows; only then
            # pool with wall lengths (valid, but the wall tail makes such intervals far too wide)
            lvl_used, method = nominal, "room-pooled"
            q = pooled_q(s, rooms, nominal)
            if not math.isfinite(q):
                lvl_used, method, q = level, "room-pooled, no PAC margin (n too small)", pooled_q(s, rooms, level)
            if not math.isfinite(q) and len(R) >= 2:
                lvl_used = len(R) / (len(R) + 1) * 0.999
                method, q = f"room-pooled, level reduced to {len(R) / (len(R) + 1):.3f} (n={len(R)})", pooled_q(s, rooms, lvl_used)
            lo = _loro(R, s, rooms, lvl_used)
            if not math.isfinite(q) and kind != "wall_length" and len(W):
                lp = _loro(R, s, rooms, nominal, pool=(sW, rW))
                if lp["loro_coverage"] is not None and lp["loro_coverage"] >= level:
                    q, lvl_used = pooled_q(np.r_[s, sW], np.r_[rooms, rW], nominal), nominal
                    method, lo = "room-pooled, shared with wall_length (too few rooms alone)", lp
            raw_cov = np.mean([abs(r["value"] - r["truth"]) <= r["raw_half"] for r in R])
            err = np.array([abs(r["value"] - r["truth"]) for r in R])
            out[tier][kind] = {
                "q": q, "level": round(min(lvl_used, level), 4), "nominal_level": round(lvl_used, 4), "method": method,
                "n": len(R), "n_rooms": int(len(np.unique(rooms))),
                **lo,
                "raw_coverage": float(raw_cov),
                "abs_err_median": float(np.median(err)),
                "width_to_error": (lo["loro_median_halfwidth"] / max(float(np.median(err)), 1e-6))
                if lo["loro_median_halfwidth"] else None,
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
