"""Concealed-damage flags and scope line items from the damage regions, by the rule tables in rules/.

Each rule's `when` is a conjunction of:
  class, surface (kind of the surface the region is on), area_m2_min / area_m2_max (region area),
  v_min_below_m / v_min_above_m (region base height above the floor; walls only), near_opening_corner_m.
Quantities carry intervals propagated from the region's area range and the surface's measurements.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ..measure import Measurement

ROOT = Path(__file__).resolve().parents[3]
SF_PER_M2 = 10.7639
FT_PER_M = 3.28084
FLOOD_CUT_M = 1.2192          # 4 ft
WICK_M = 0.33                 # EPA wicking test: moisture ~33 cm above a ~8 cm visible line


def _rules(name: str) -> list[dict]:
    return yaml.safe_load((ROOT / "rules" / f"{name}.yaml").read_text())


def _surface_info(result: dict) -> dict:
    """surface id -> kind, area measurement, and for walls the wall json and its room's openings."""
    walls = {w["id"]: (w, rm) for rm in result["rooms"] for w in rm["walls"]}
    out = {}
    for s in result["surfaces"]:
        w, rm = walls.get(s.get("wall_id"), (None, None))
        out[s["id"]] = {"kind": s["kind"], "area": s["area"], "wall": w,
                        "openings": [o for o in (rm or {}).get("openings", []) if w and o["wall_id"] == w["id"]]}
    return out


def _near_opening_corner(r: dict, info: dict, dist: float) -> bool:
    e = r["extent"]
    for o in info["openings"]:
        u0 = o["offset"]["value"]
        u1 = u0 + o["width"]["value"]
        v0 = o.get("sill", {}).get("value", 0.0)
        v1 = v0 + o["height"]["value"]
        for cu in (u0, u1):
            for cv in (v0, v1):
                du = max(e["u_min"] - cu, 0.0, cu - e["u_max"])
                dv = max(e["v_min"] - cv, 0.0, cv - e["v_max"])
                if np.hypot(du, dv) <= dist:
                    return True
    return False


def _matches(when: dict, r: dict, info: dict) -> bool:
    a = r["area"]["value"]
    if "class" in when and r["class"] not in when["class"]:
        return False
    if "surface" in when and info["kind"] not in when["surface"]:
        return False
    if "area_m2_min" in when and a < when["area_m2_min"]:
        return False
    if "area_m2_max" in when and a >= when["area_m2_max"]:
        return False
    if "v_min_below_m" in when and not (info["kind"] == "wall" and r["extent"]["v_min"] < when["v_min_below_m"]):
        return False
    if "v_min_above_m" in when and info["kind"] == "wall" and r["extent"]["v_min"] < when["v_min_above_m"]:
        return False
    if "near_opening_corner_m" in when and not _near_opening_corner(r, info, when["near_opening_corner_m"]):
        return False
    return True


def _quantity(kind: str, r: dict, info: dict) -> Measurement:
    ar = r["area"]
    method = f"rule:{kind} (from region and surface intervals)"
    if kind == "region_area":
        return Measurement(ar["value"] * SF_PER_M2, ar["lo"] * SF_PER_M2, ar["hi"] * SF_PER_M2, unit="ft2", method=method)
    if kind == "wall_area":
        s = info["area"]
        return Measurement(s["value"] * SF_PER_M2, s["lo"] * SF_PER_M2, s["hi"] * SF_PER_M2, unit="ft2", method=method)
    if kind == "flood_cut":
        L = info["wall"]["length"]
        cut = max(FLOOD_CUT_M, r["extent"]["v_max"] + WICK_M)
        return Measurement(L["value"] * cut * SF_PER_M2, L["lo"] * cut * SF_PER_M2, L["hi"] * cut * SF_PER_M2,
                           unit="ft2", method=method)
    if kind == "crack_length":
        e = r["extent"]
        d = float(np.hypot(e["u_max"] - e["u_min"], e["v_max"] - e["v_min"]))
        side = max(e["u_max"] - e["u_min"], e["v_max"] - e["v_min"])        # a straight crack along one axis
        return Measurement(d * FT_PER_M, side * FT_PER_M, d * FT_PER_M, unit="ft", method=method)
    if kind == "count":
        return Measurement(1.0, 1.0, 1.0, unit="count", method=method)
    raise ValueError(f"unknown quantity {kind}")


def apply(result: dict) -> tuple[list[dict], list[dict]]:
    """(concealed_flags, scope_items) for the result's damage regions."""
    info = _surface_info(result)
    flags, items = [], []
    for r in result["damage_regions"]:
        si = info.get(r["surface_id"])
        if si is None:
            continue
        inputs = {"region": r["id"], "class": r["class"], "surface": si["kind"], "area_m2": r["area"]["value"],
                  "base_height_m": r["extent"]["v_min"] if si["kind"] == "wall" else None}
        for rule in _rules("concealed"):
            if _matches(rule["when"], r, si):
                flags.append({"id": f"C{len(flags) + 1}", "surface_id": r["surface_id"], "rule_id": rule["id"],
                              "rule": rule["rule"], "inputs": inputs, "action": rule["action"],
                              "citation": rule["citation"]})
        for rule in _rules("scope"):
            if _matches(rule["when"], r, si):
                key = (rule["id"], r["surface_id"])
                prev = next((x for x in items if (x["rule_id"], x["surface_id"]) == key), None)
                if prev is not None and rule["quantity"] in ("wall_area", "flood_cut"):
                    prev["triggered_by"].append(r["id"])      # one whole-surface item, however many regions
                    continue
                items.append({"id": f"SC{len(items) + 1}", "surface_id": r["surface_id"], "code": rule["code"],
                              "description": rule["description"], "unit": rule["unit"],
                              "quantity": _quantity(rule["quantity"], r, si).to_json(), "rule_id": rule["id"],
                              "triggered_by": [r["id"]]})
    return flags, items
