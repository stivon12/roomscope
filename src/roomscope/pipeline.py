"""Orchestration: capture -> front-end (tier specific) -> shared core -> result.json + plan.png."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from . import __version__
from .core import layout as L
from .measure import Measurement


ROOT = Path(__file__).resolve().parents[2]
UNOBSERVED_FRAC = 0.3       # wall length fraction below which a wall counts as unobserved
UNOBSERVED_POS_SE = 0.05    # m, position standard error of an inferred wall


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _name_room(poly: Polygon, k: int) -> str:
    minx, miny, maxx, maxy = poly.bounds
    w, h = sorted((maxx - minx, maxy - miny))
    return "Hallway" if w < 1.6 and h / max(w, 1e-6) > 2.5 else f"Room {k + 1}"


def analyse(cap, corrs, warnings: list[str], single_room: bool = False):
    """Shared geometry core on posed frames. Returns plain-python room/opening structures."""
    # every view counts when there are few (photos: 2-8, video keyframes: ~32); only dense LiDAR streams
    # (hundreds of frames at 10+ Hz, consecutive frames nearly identical) are thinned
    step = 1 if len(cap.poses) <= 100 else 2
    cloud = L.fuse(cap, corrs, frame_step=step)
    fp = cloud.P[cloud.cls == "floor"]
    floor = L.fit_hplane(fp, z0=L.pick_level(fp, "low"))
    cp = cloud.P[cloud.cls == "ceil"]
    ccfg = L.layout_config()["ceiling"]
    # no ceiling seen anywhere: fit walls up to the prior maximum instead of assuming a height
    ceil_z = (L.pick_level(cp, "high") - floor.c) if len(cp) > 100 else ccfg["max_m"]
    on_wall, scfg = None, L.layout_config().get("semantics", {})
    mode = os.environ.get("ROOMSCOPE_SEM_WALLS", scfg.get("wall_mode", "off"))
    if scfg.get("enabled") and mode != "off" and getattr(cap, "depth_hw", None) is not None:
        from .core import semantics as S
        try:
            frames = S.label_frames(cap, ROOT / "out" / "cache" / "seg", scfg["interval_s"])
            cloud.sem, _ = S.label_cloud(cloud, cap, corrs, frames, scfg["radius_m"])
            on_wall = S.on_wall_mask(cloud.sem, S.label_frames.names, scfg["on_wall_classes"])
        except Exception as e:                # labels are an aid: geometry alone still produces a plan
            warnings.append(f"semantic labels unavailable ({type(e).__name__}: {e}); walls judged by geometry only")
    walls = L.fit_wall_planes(cloud, floor, ceil_z, on_wall=on_wall, sem_mode=mode,
                              min_labelled=scfg.get("min_labelled", 0.3))
    masks, g, cuts, inferred = L.segment_rooms(cloud, floor, walls, single_room=single_room)
    for a in getattr(L.segment_rooms, "not_entered", []):
        warnings.append(f"a {a:.1f} m2 region was seen (e.g. through a doorway) but the camera never walked into it: "
                        "not reported as a room")

    polys = []
    for k, (m, inf_area) in enumerate(zip(masks, inferred)):
        if inf_area > 0.25:
            warnings.append(f"{inf_area:.2f} m2 of floor inferred (enclosed by walls, not directly observed)")
        others = np.any([o for j, o in enumerate(masks) if j != k], axis=0) if len(masks) > 1 else None
        r = L.room_polygon(m, g, walls, others)
        if r is None:
            warnings.append("a floor region could not be turned into a closed polygon and was dropped")
            continue
        polys.append(r)
    polys, rw = L.reconcile_rooms(polys)
    warnings += rw
    rooms = []
    for corners, edges in polys:
        poly = Polygon(corners)
        ch = L.ceiling_height(cloud, floor, poly)
        rooms.append({"corners": corners, "edges": edges, "poly": poly, "ceil": ch})
    _fill_unobserved_ceilings(rooms, walls, ccfg, warnings, cloud.cams - [0, 0, floor.c])

    faces = []
    for ri, r in enumerate(rooms):
        n = len(r["corners"])
        for k, e in enumerate(r["edges"]):
            p, q = np.array(r["corners"][k]), np.array(r["corners"][(k + 1) % n])
            a, b = sorted((p[1 - e.axis], q[1 - e.axis]))
            sign = 1 if e.face[0] == "+" else -1
            faces.append(L.Face(ri, k, e.axis, sign, e.c, a, b, r["ceil"][0],
                                observed=e.plane is not None and e.plane.kind == "wall"))
    R = cloud.R
    grids = L.opening_evidence(cap, lambda i: R @ corrs[i], faces, [r["poly"] for r in rooms], floor,
                               frame_step=step)
    openings = L.detect_openings(grids, faces, cloud.cams[:, :2])
    cloud.floor = floor                                   # for surface-anchored outputs (damage)
    cloud.frame_to_result = lambda i: R @ corrs[i]
    return rooms, faces, grids, openings, cloud


def _fill_unobserved_ceilings(rooms, walls, ccfg: dict, warnings: list[str], cams: np.ndarray):
    """Rooms whose ceiling was never scanned get a range, flagged observed: false, never a measurement:
    - other rooms of the capture measured theirs: their median, widened by their spread and the
      room-to-room prior (config/layout.yaml ceiling.room_sigma_m);
    - nobody saw a ceiling: from the highest wall point near the room or the highest camera position
      inside it (the ceiling is above both; cams are heights above the floor) to the residential maximum, centred on the typical height clamped into that range."""
    seen = [r["ceil"][0] for r in rooms if r["ceil"] is not None]
    for ri, r in enumerate(rooms):
        if r["ceil"] is not None:
            r["ceil_meas"] = None
            continue
        if seen:
            v = float(np.median(seen))
            half = L.Z95 * float(np.hypot(np.std(seen), ccfg["room_sigma_m"]))
            m = Measurement(v, v - half, v + half, method="inferred:other-rooms-of-capture", observed=False)
            warnings.append(f"R{ri + 1} ceiling not observed: {m.lo:.2f}-{m.hi:.2f} m inferred from the other rooms, "
                            "not a measurement")
        else:
            near = r["poly"].buffer(0.3)
            tops = [w.top for w in walls if w.kind == "wall" and any(
                near.intersects(_wall_line(w, lo, hi)) for lo, hi in w.segments)]
            inside = [z for x, y, z in cams if r["poly"].contains(Point(x, y))]
            lo = max(tops + inside) if tops or inside else 0.0
            src = "highest wall point" if tops and max(tops) >= max(inside, default=0.0) else "highest camera position"
            hi = max(ccfg["max_m"], lo)
            v = float(np.clip(ccfg["typical_m"], lo, hi))
            m = Measurement(v, lo, hi, method="not-observed:residential-prior", observed=False)
            warnings.append(f"R{ri + 1} ceiling not observed: at least {lo:.2f} m ({src}), reported range "
                            f"{lo:.2f}-{hi:.2f} m, not a measurement")
        r["ceil"] = (m.value, (m.hi - m.lo) / (2 * L.Z95), 0)
        r["ceil_meas"] = m


def _wall_line(w, lo: float, hi: float):
    from shapely.geometry import LineString
    return LineString([(w.offset, lo), (w.offset, hi)] if w.axis == 0 else [(lo, w.offset), (hi, w.offset)])


# An open boundary is where the partition put two rooms' floors side by side with no wall between them:
# the passage is certain (both floors were walked or seen), its exact extent is not (it is the partition
# line, not a detected door frame). Reported at an even confidence rather than an invented one.
SHARED_BOUNDARY_CONF = 0.5


def _opening_height(o, wall_h: float) -> dict:
    """Measured when wall was seen above the opening; otherwise a range flagged observed: false, from
    the top of what was seen through it up to the wall height, centred on the standard door height
    (doors) or the middle of that range (windows)."""
    h = o.v1 - o.v0
    if o.head_observed:
        return L.meas_len(h, o.height_se).to_json()
    lo, hi = h, max(h, wall_h - o.v0)
    typical = L.layout_config().get("openings", {}).get("door_head_typical_m", 2.03) - o.v0
    v = float(np.clip(typical, lo, hi)) if o.type != "window" else 0.5 * (lo + hi)
    return Measurement(v, lo, hi, method="not-observed:top-unseen", observed=False).to_json()


def assemble(rooms, faces, grids, openings, warnings) -> dict:
    out_rooms, surfaces, adjacency = [], [], []
    shared_links = []        # (room, neighbour, opening id) from open boundaries of the partition
    op_records = []          # (room_id, opening_json, OpeningEst)
    for ri, r in enumerate(rooms):
        rid = f"R{ri + 1}"
        corners, edges, poly = r["corners"], r["edges"], r["poly"]
        h, h_se, _ = r["ceil"]
        height = r.get("ceil_meas") or L.meas_len(h, h_se)
        n = len(corners)
        walls_json = []
        area_var = 0.0
        # observed fraction = share of the wall's LENGTH with direct returns (columns with >= 3 solid hits);
        # the upper part of a wall is rarely scanned, so an area fraction would call every wall unobserved
        seen_frac, pos_se = [], []
        for k, e in enumerate(edges):
            fi = next(i for i, f in enumerate(faces) if f.room == ri and f.wall_idx == k)
            S = grids[fi]["solid"]
            seen = float((S.sum(0) >= 3).mean()) if S.size else 0.0
            seen_frac.append(seen)
            # where the wall was not seen, its position is inferred (extended line or floor boundary):
            # +-5 cm position uncertainty instead of the plane-fit standard error
            pos_se.append(e.plane.se if (e.plane is not None and seen >= UNOBSERVED_FRAC) else UNOBSERVED_POS_SE)
            if seen < UNOBSERVED_FRAC:
                warnings.append(f"{rid}-W{k + 1}: only {seen:.0%} of its length observed; position inferred, interval widened")
        for k, e in enumerate(edges):
            p, q = corners[k], corners[(k + 1) % n]
            length = float(np.hypot(q[0] - p[0], q[1] - p[1]))
            se = np.hypot(pos_se[k - 1], pos_se[(k + 1) % n])
            seen = seen_frac[k]
            walls_json.append({
                "id": f"{rid}-W{k + 1}", "start": [round(p[0], 4), round(p[1], 4)], "end": [round(q[0], 4), round(q[1], 4)],
                "length": L.meas_len(length, se).to_json(), "height": height.to_json(),
                "observed_fraction": round(float(seen), 3),
            })
            area_var += (length * pos_se[k]) ** 2
            if e.shared:
                warnings.append(f"{rid}-W{k + 1}: open boundary with another room (no wall), position from the room partition")
            elif e.plane is None:
                warnings.append(f"{rid}-W{k + 1}: no fitted wall plane, edge taken from floor boundary")
        area = poly.area
        a_half = L.Z95 * np.sqrt(area_var + (poly.length * L.SYS_LEN) ** 2)
        ops_json = []
        for oi, o in enumerate([o for o in openings if o.face.room == ri]):
            f = o.face
            wall = walls_json[f.wall_idx]
            s_u = wall["start"][1 - f.axis]
            e_u = wall["end"][1 - f.axis]
            offset = (o.u0 - s_u) if e_u > s_u else (s_u - o.u1)
            plane = edges[f.wall_idx].plane
            oj = {
                "id": f"{rid}-O{oi + 1}", "type": o.type, "wall_id": wall["id"],
                "width": L.meas_len(o.u1 - o.u0, o.width_se).to_json(),
                "height": _opening_height(o, height.value),
                "offset": L.meas_len(offset, np.hypot(o.width_se, plane.se if plane else 0.03)).to_json(),
                "confidence": round(o.conf, 3),
            }
            if o.type == "window":
                oj["sill"] = L.meas_len(o.v0, o.height_se).to_json()
            if not o.head_observed:
                hj = oj["height"]
                warnings.append(f"{oj['id']} ({o.type}): top not seen, height reported as {hj['lo']:.2f}-{hj['hi']:.2f} m, "
                                "not a measurement")
            ops_json.append(oj)
            op_records.append((rid, oj, o))
        # an open boundary with another room (no wall: doorway or open junction found by the partition) is
        # a full-width, full-height opening onto that room
        for k, e in enumerate(edges):
            if not e.shared:
                continue
            p, q = corners[k], corners[(k + 1) % n]
            mid = Point((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            nbr = [j for j, o in enumerate(rooms) if j != ri and o["poly"].distance(mid) < 0.1]
            if not nbr:
                continue
            se = np.hypot(pos_se[k - 1], pos_se[(k + 1) % n])
            oj = {"id": f"{rid}-O{len(ops_json) + 1}", "type": "opening", "wall_id": walls_json[k]["id"],
                  "width": L.meas_len(float(np.hypot(q[0] - p[0], q[1] - p[1])), se).to_json(),
                  "height": height.to_json(), "offset": L.meas_len(0.0, 0.0).to_json(),
                  "confidence": SHARED_BOUNDARY_CONF}
            ops_json.append(oj)
            shared_links.append((rid, f"R{nbr[0] + 1}", oj["id"]))
        name = _name_room(poly, ri)
        out_rooms.append({
            "id": rid, "name": name, "polygon": [[round(x, 4), round(y, 4)] for x, y in corners],
            "walls": walls_json, "openings": ops_json, "ceiling_height": height.to_json(),
            "floor_area": Measurement.from_abs(area, a_half, unit="m2", method="fitstat:v0").to_json(),
            "perimeter": L.meas_len(poly.length, np.sqrt(sum((w["length"]["hi"] - w["length"]["value"]) ** 2 for w in walls_json)) / L.Z95).to_json(),
        })
        for w in walls_json:
            op_area = sum(oj["width"]["value"] * oj["height"]["value"] for oj in ops_json if oj["wall_id"] == w["id"])
            a = w["length"]["value"] * height.value - op_area
            rel = np.hypot((w["length"]["hi"] - w["length"]["value"]) / max(w["length"]["value"], 1e-6),
                           (height.hi - height.value) / max(height.value, 1e-6))
            surfaces.append({"id": f"S-{w['id']}", "room_id": rid, "kind": "wall", "wall_id": w["id"],
                             "area": Measurement.from_rel(a, rel, unit="m2", method="fitstat:v0").to_json()})
        fa = out_rooms[-1]["floor_area"]
        surfaces.append({"id": f"S-{rid}-floor", "room_id": rid, "kind": "floor", "area": fa})
        surfaces.append({"id": f"S-{rid}-ceiling", "room_id": rid, "kind": "ceiling", "area": fa})

    # adjacency: the same doorway seen from both sides (parallel walls, facing opposite, <0.4 m apart, overlapping)
    for i in range(len(op_records)):
        for j in range(i + 1, len(op_records)):
            ra, ja, a = op_records[i]
            rb, jb, b = op_records[j]
            if ra == rb or a.face.axis != b.face.axis or a.face.sign == b.face.sign:
                continue
            if abs(a.face.offset - b.face.offset) > 0.4:
                continue
            ov = min(a.u1, b.u1) - max(a.u0, b.u0)
            if ov < 0.5 * min(a.u1 - a.u0, b.u1 - b.u0):
                continue
            conf = float(np.clip(ov / max(a.u1 - a.u0, b.u1 - b.u0), 0, 1) * min(a.conf, b.conf) ** 0.5)
            adjacency.append({"room_a": ra, "room_b": rb, "via": ja["id"], "confidence": round(conf, 3)})

    linked = {frozenset((a["room_a"], a["room_b"])) for a in adjacency}
    # a door or opening seen from one side only leads to the room whose floor lies just beyond it
    for ra, oj, o in op_records:
        if o.type == "window":
            continue
        f = o.face
        um = (o.u0 + o.u1) / 2
        for depth in (0.3, 0.6):
            xy = [0.0, 0.0]
            xy[f.axis], xy[1 - f.axis] = f.offset - f.sign * depth, um
            hit = [j for j, r in enumerate(rooms) if f"R{j + 1}" != ra and r["poly"].contains(Point(*xy))]
            if hit:
                rb = f"R{hit[0] + 1}"
                if frozenset((ra, rb)) not in linked:
                    linked.add(frozenset((ra, rb)))
                    adjacency.append({"room_a": ra, "room_b": rb, "via": oj["id"], "confidence": round(o.conf, 3)})
                break
    for ra, rb, oid in shared_links:
        if frozenset((ra, rb)) not in linked:
            linked.add(frozenset((ra, rb)))
            adjacency.append({"room_a": ra, "room_b": rb, "via": oid, "confidence": SHARED_BOUNDARY_CONF})

    polys = [r["poly"] for r in rooms]
    union = unary_union(polys)
    overlap = sum(polys[i].intersection(polys[j]).area for i in range(len(polys)) for j in range(i + 1, len(polys)))
    total = sum(p.area for p in polys)
    f_half = np.sqrt(sum(((r["floor_area"]["hi"] - r["floor_area"]["value"])) ** 2 for r in out_rooms))
    hull = union.convex_hull if union.geom_type != "Polygon" else union
    footprint = {
        "area": Measurement.from_abs(total, f_half, unit="m2", method="fitstat:v0").to_json(),
        "polygon": ([[round(x, 4), round(y, 4)] for x, y in list(hull.exterior.coords)[:-1]]
                    if hull.geom_type == "Polygon" else []),          # no room reconstructed: empty
        "stitch_status": "single_room" if len(rooms) == 1 else "stitched",
        "max_room_overlap_m2": round(float(overlap), 4),
    }
    return {"rooms": out_rooms, "adjacency": adjacency, "footprint": footprint, "surfaces": surfaces}


def _geometry(cap, drift: bool, warnings: list[str], single_room: bool = False):
    """Shared core on one posed capture: drift correction (or the ablation), layout, assembly."""
    from .core import drift as D
    if drift:
        dr = D.correct_drift(cap)
        corrs = dr.corrections
        drift_meta = {"enabled": True, "method": dr.method, "residual_before_m": round(dr.residual_before, 4),
                      "residual_after_m": round(dr.residual_after, 4), "fragments": dr.n_fragments}
    else:
        corrs = D.identity(cap)
        drift_meta = {"enabled": False, "method": "none (ablation: odometry poses used as-is)"}
    rooms, faces, grids, openings, cloud = analyse(cap, corrs, warnings, single_room=single_room)
    return assemble(rooms, faces, grids, openings, warnings), cloud, drift_meta


def _rename_room(obj, old: str, new: str):
    """Recursively rename a room id inside its JSON (R1 -> R3, R1-W2 -> R3-W2, S-R1-floor -> S-R3-floor)."""
    if isinstance(obj, dict):
        return {k: _rename_room(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rename_room(v, old, new) for v in obj]
    if isinstance(obj, str):
        if obj == old:
            return new
        for pre in (old + "-", "S-" + old + "-"):
            if obj.startswith(pre):
                return obj.replace(old, new, 1)
    return obj


def _save_cloud(out: Path, cloud, align, name: str = "cloud.npz"):
    keep = np.unique(np.floor(cloud.P / 0.05).astype(np.int64), axis=0, return_index=True)[1]
    np.savez_compressed(out / name, P=cloud.P[keep].astype(np.float32), N=cloud.N[keep].astype(np.float32),
                        T_world_to_result=cloud.R @ align)


def run_capture(capture: Path, tier: str, out_dir: Path, drift: bool = True, load_kw: dict | None = None,
                depth_scale: float | None = None, depth_correction: bool = True, device: str | None = None,
                calibrate: bool = True, n_frames: int = 32, damage: bool = True) -> Path:
    t0 = time.time()
    capture = Path(capture)
    out = Path(out_dir) / capture.name
    out.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    models: list[dict] = []
    ds_meta = {"enabled": False, "source": f"not applicable to the {tier} tier"}
    clouds = []                                  # (cloud, align, filename)

    if tier == "lidar":
        from .core.depth_calib import resolve_depth_scale
        from .frontends.lidar import load_any
        load_kw = dict(load_kw or {})
        if "depth_affine" in load_kw:              # diagnostics pass an explicit model
            a, b = load_kw["depth_affine"]
            ds_meta = {"enabled": True, "scale": a, "offset_m": b, "se": 0.0, "source": "explicit depth_affine"}
        else:
            from .frontends.lidar import device_of
            device = device or device_of(capture)
            ds = resolve_depth_scale(device, depth_scale, depth_correction)
            load_kw["depth_affine"] = None if ds.scale == 1.0 else (ds.scale, 0.0)
            ds_meta = ds.to_json()
            if depth_correction and depth_scale is None and ds.device == "unknown":
                warnings.append(f"device {'not given' if not device else repr(device) + ' has no measured depth scale'}: "
                                f"LiDAR scale uncertainty {100 * ds.se:.1f}% added to every interval "
                                "(pass --device, or measure one distance with `roomscope calibrate-depth`)")
        cap = load_any(capture, **load_kw)
        warnings += getattr(cap, "load_warnings", [])
        body, cloud, drift_meta = _geometry(cap, drift, warnings)
        if cloud.sem is not None:
            from .core.semantics import MODEL_INFO as SEG_INFO
            models.append(SEG_INFO)
        if damage:
            from . import damage as Dm
            body["damage_regions"], dm_model = Dm.detect(cap, cloud.frame_to_result, body, cloud.floor.z,
                                                         ROOT / "out" / "cache" / "damage", warnings)
            if dm_model:
                models.append(dm_model)
        clouds.append((cloud, cap.align, "cloud.npz"))
    elif tier == "video":
        from .frontends.recon import MODEL_INFO
        from .frontends.video import load_video
        cap = load_video(capture, out, n_frames=n_frames)
        models.append(MODEL_INFO)
        body, cloud, drift_meta = _geometry(cap, drift, warnings)
        clouds.append((cloud, cap.align, "cloud.npz"))
    elif tier == "photo":
        from .core import stitch as St
        from .frontends.photo import load_room, room_folders
        from .frontends.recon import MODEL_INFO
        models.append(MODEL_INFO)
        drift_meta = {"enabled": False, "method": "not applicable: photos have no trajectory to drift"}
        per_room = []
        for k, folder in enumerate(room_folders(capture)):
            w: list[str] = []
            try:
                cap = load_room(folder, work=out / "rooms" / folder.name)
                b, cloud, _ = _geometry(cap, False, w, single_room=True)
            except Exception as e:                    # one bad folder must not sink the whole property
                warnings.append(f"{folder.name}: skipped, {type(e).__name__}: {e}")
                continue
            if not b["rooms"]:
                warnings.append(f"{folder.name}: no room could be reconstructed from its photos")
                continue
            main = max(b["rooms"], key=lambda r: r["floor_area"]["value"])
            if len(b["rooms"]) > 1:
                w.append(f"{folder.name}: {len(b['rooms'])} floor regions found, kept the largest")
            rid = f"R{len(per_room) + 1}"
            room = _rename_room(main, main["id"], rid)
            room["name"] = folder.name
            surf = [_rename_room(s, main["id"], rid) for s in b["surfaces"] if s["room_id"] == main["id"]]
            per_room.append((room, surf))
            warnings += [_rename_room(x, main["id"], rid) for x in w]
            clouds.append((cloud, cap.align, "cloud.npz" if k == 0 else f"cloud_{rid}.npz"))
        if not per_room:
            raise RuntimeError(f"no room reconstructed from {capture}")
        placed, adjacency, status, sw = St.stitch([r for r, _ in per_room])
        warnings += sw
        body = {"rooms": placed, "adjacency": adjacency, "footprint": St.footprint(placed, status),
                "surfaces": [s for _, ss in per_room for s in ss]}
    else:
        raise ValueError(f"unknown tier {tier}")

    result = {
        "schema_version": "1.0",
        "meta": {
            "capture_id": capture.name, "tier": tier, "device": device or "unknown", "os_version": "unknown",
            "pipeline_version": __version__, "git_commit": _git_commit(), "models": models,
            "drift_correction": drift_meta, "depth_correction": ds_meta, "runtime_s": round(time.time() - t0, 2),
        },
        "damage_regions": [], **body, "concealed_flags": [], "scope_items": [],
        "warnings": sorted(set(warnings)),
    }
    if calibrate:
        from .core import calibrate as Cal
        result = Cal.apply(result, tier)
        result["warnings"] = sorted(set(result["warnings"]))
    from . import scope as Sc                    # after calibration: quantities use the calibrated surfaces
    result["concealed_flags"], result["scope_items"] = Sc.apply(result)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    for cloud, align, name in clouds:            # fused clouds in the result frame (5 cm), for GT scoring
        _save_cloud(out, cloud, align, name)
    from .render import render_plan
    render_plan(result, out / "plan.png")
    return out / "result.json"
