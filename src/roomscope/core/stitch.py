"""Photo tier: place independently reconstructed rooms into one property frame.

Each room folder is reconstructed alone (its own origin and heading, gravity-aligned and squared to
its walls). Rooms are linked through doors. A door seen from both sides has the same width, sits on
walls facing opposite ways, and its two faces are one wall thickness apart (about 12 cm, interior
partition). For a placed room A and an unplaced room B, every (door in A, door in B, rotation of B by
k*90 deg) that makes the two door walls face each other gives a placement hypothesis. B's door centre
lands on A's door centre pushed through the wall.

Scoring:
- width disagreement (cm);
- polygon overlap with every room already placed (m^2), heavily penalised: rooms cannot overlap;
- door-wall collinearity is enforced by construction.

Rooms are added greedily, best hypothesis first, starting from the largest room. A room with no
acceptable hypothesis is NOT guessed into place: it is laid out to the side, stitch_status becomes
"partial", and a warning names it. A wrong confident stitch is worse than an honest gap.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

WALL_T = 0.12          # m, assumed partition thickness between the two faces of a doorway
MAX_DW = 0.20          # m, max door width disagreement
MAX_OVERLAP = 0.25     # m^2, max overlap with placed rooms for an acceptable placement


def _rot(k: int) -> np.ndarray:
    c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k % 4]
    return np.array([[c, -s], [s, c]], float)


def _doors(room: dict) -> list[dict]:
    """Door centre (2-D), outward normal and width, from the room JSON."""
    walls = {w["id"]: w for w in room["walls"]}
    poly = Polygon(room["polygon"])
    out = []
    for o in room["openings"]:
        if o["type"] not in ("door", "opening"):
            continue
        w = walls[o["wall_id"]]
        a, b = np.array(w["start"]), np.array(w["end"])
        t = (b - a) / np.linalg.norm(b - a)
        c = a + t * (o["offset"]["value"] + o["width"]["value"] / 2)
        n = np.array([t[1], -t[0]])
        if poly.contains(Point(*(c + 0.10 * n))):
            n = -n                                   # make n point OUT of the room
        out.append({"id": o["id"], "c": c, "n": n, "w": o["width"]["value"]})
    return out


def _transform_room(room: dict, R: np.ndarray, t: np.ndarray) -> dict:
    r = copy.deepcopy(room)
    f = lambda p: (R @ np.asarray(p, float) + t).round(4).tolist()
    r["polygon"] = [f(p) for p in r["polygon"]]
    for w in r["walls"]:
        w["start"], w["end"] = f(w["start"]), f(w["end"])
    return r


def stitch(rooms: list[dict]) -> tuple[list[dict], list[dict], str, list[str]]:
    """rooms: per-room JSON (each in its own frame). Returns (placed rooms, adjacency, status, warnings)."""
    warnings = []
    if len(rooms) == 1:
        return rooms, [], "single_room", warnings
    order = sorted(range(len(rooms)), key=lambda i: -Polygon(rooms[i]["polygon"]).area)
    placed = {order[0]: rooms[order[0]]}
    adjacency = []
    pending = set(order[1:])
    while pending:
        best = None
        for i in pending:
            dB = _doors(rooms[i])
            for j, A in placed.items():
                for da in _doors(A):
                    for db in dB:
                        if abs(da["w"] - db["w"]) > MAX_DW:
                            continue
                        for k in range(4):
                            R = _rot(k)
                            if R @ db["n"] @ da["n"] > -0.99:          # door walls must face each other
                                continue
                            t = da["c"] + da["n"] * WALL_T - R @ db["c"]
                            cand = _transform_room(rooms[i], R, t)
                            P = Polygon(cand["polygon"])
                            ov = sum(P.intersection(Polygon(q["polygon"])).area for q in placed.values())
                            score = ov * 10 + abs(da["w"] - db["w"]) * 10
                            if ov <= MAX_OVERLAP and (best is None or score < best[0]):
                                conf = max(0.0, 1 - abs(da["w"] - db["w"]) / MAX_DW) * max(0.0, 1 - ov / MAX_OVERLAP)
                                best = (score, i, cand, A["id"], da["id"], db["id"], conf)
        if best is None:
            break
        _, i, cand, a_id, da_id, db_id, conf = best
        placed[i] = cand
        pending.discard(i)
        adjacency.append({"room_a": a_id, "room_b": cand["id"], "via": da_id, "confidence": round(conf, 3)})
    status = "stitched"
    if pending:
        status = "partial"
        x0 = max(max(p[0] for p in r["polygon"]) for r in placed.values()) + 2.0
        for i in sorted(pending):
            P = np.asarray(rooms[i]["polygon"])
            placed[i] = _transform_room(rooms[i], np.eye(2), np.array([x0 - P[:, 0].min(), -P[:, 1].min()]))
            x0 = max(p[0] for p in placed[i]["polygon"]) + 2.0
            warnings.append(f"{rooms[i]['id']} ({rooms[i]['name']}): no door matched another room; placed aside, "
                            f"not stitched (adjacency unknown)")
    return [placed[i] for i in range(len(rooms))], adjacency, status, warnings


def footprint(rooms: list[dict], status: str) -> dict:
    polys = [Polygon(r["polygon"]) for r in rooms]
    union = unary_union(polys)
    overlap = sum(polys[i].intersection(polys[j]).area for i in range(len(polys)) for j in range(i + 1, len(polys)))
    halves = [r["floor_area"]["hi"] - r["floor_area"]["value"] for r in rooms]
    tot = sum(p.area for p in polys)
    h = math.sqrt(sum(x * x for x in halves))
    hull = union.convex_hull if union.geom_type != "Polygon" else union
    return {"area": {"value": round(tot, 4), "lo": round(tot - h, 4), "hi": round(tot + h, 4), "unit": "m2",
                     "level": rooms[0]["floor_area"]["level"], "method": rooms[0]["floor_area"]["method"]},
            "polygon": ([[round(x, 4), round(y, 4)] for x, y in list(hull.exterior.coords)[:-1]]
                    if hull.geom_type == "Polygon" else []),          # no room reconstructed: empty
            "stitch_status": status, "max_room_overlap_m2": round(float(overlap), 4)}
