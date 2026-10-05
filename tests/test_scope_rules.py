"""Rule engine (src/roomscope/scope): which flags and scope items fire for a region. Hand-written region
records exercise the rule logic only; they are not benchmark data."""
from roomscope import scope as Sc


def _m(v, lo=None, hi=None, unit="m2"):
    return {"value": v, "lo": v if lo is None else lo, "hi": v if hi is None else hi, "unit": unit, "level": 0.9}


def _result(regions, openings=()):
    wall = {"id": "R1-W1", "start": [0, 0], "end": [4, 0], "length": _m(4.0, 3.95, 4.05, "m")}
    return {"rooms": [{"id": "R1", "walls": [wall], "openings": list(openings)}],
            "surfaces": [{"id": "S-R1-W1", "room_id": "R1", "kind": "wall", "wall_id": "R1-W1", "area": _m(10.0, 9.5, 10.5)},
                         {"id": "S-R1-ceiling", "room_id": "R1", "kind": "ceiling", "area": _m(12.0)}],
            "damage_regions": regions}


def _region(cls, surface="S-R1-W1", area=0.05, v=(1.0, 1.2), u=(1.0, 1.2)):
    return {"id": "D1", "surface_id": surface, "class": cls, "area": _m(area, 0.8 * area, 1.2 * area),
            "extent": {"u_min": u[0], "u_max": u[1], "v_min": v[0], "v_max": v[1]}, "confidence": 0.5}


def ids(xs):
    return sorted(x["rule_id"] for x in xs)


def test_ceiling_stain_flags_cavity_and_seals():
    f, s = Sc.apply(_result([_region("water_stain", "S-R1-ceiling")]))
    assert ids(f) == ["F4-ceiling-water"]
    assert ids(s) == ["S2-stain-seal-paint"]


def test_low_wall_stain_is_wicking_with_flood_cut_above_line():
    f, s = Sc.apply(_result([_region("water_stain", v=(0.02, 1.1))]))
    assert "F6-wicking" in ids(f)
    cut = next(x for x in s if x["rule_id"] == "S1-flood-cut")
    assert abs(cut["quantity"]["value"] - 4.0 * (1.1 + 0.33) * 10.7639) < 0.01   # above 4 ft: line + wicking
    assert "S2-stain-seal-paint" not in ids(s)


def test_mold_size_tiers():
    assert ids(Sc.apply(_result([_region("mold", area=0.5)]))[0]) == ["F1-mold-small"]
    assert ids(Sc.apply(_result([_region("mold", area=2.0)]))[0]) == ["F2-mold-medium"]
    assert ids(Sc.apply(_result([_region("mold", area=12.0)]))[0]) == ["F3-mold-large"]


def test_crack_at_door_corner():
    door = {"id": "R1-O1", "type": "door", "wall_id": "R1-W1", "offset": _m(2.0, unit="m"), "width": _m(0.9, unit="m"),
            "height": _m(2.0, unit="m")}
    f, _ = Sc.apply(_result([_region("crack", v=(2.0, 2.4), u=(2.9, 3.1))], [door]))
    assert "F8-crack-at-opening" in ids(f)
    f, _ = Sc.apply(_result([_region("crack", v=(0.5, 0.7), u=(0.2, 0.4))], [door]))
    assert "F8-crack-at-opening" not in ids(f)


def test_one_whole_wall_item_for_several_regions():
    a, b = _region("hole"), _region("hole")
    b["id"] = "D2"
    _, s = Sc.apply(_result([a, b]))
    assert ids(s).count("S6-hole-patch") == 2                                        # one patch per hole
    paint = [x for x in s if x["rule_id"] == "S8-repaint-after-repair"]
    assert len(paint) == 1 and paint[0]["triggered_by"] == ["D1", "D2"]
