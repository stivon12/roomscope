"""Shared geometry core: posed points -> rooms (Manhattan polygons), walls, ceiling heights, openings.

Every tier feeds this module the same thing (per-frame posed points + normals); intervals widen with
the front-end's noise because they are computed from the fit residuals.

Pipeline, and why each step is the simple explainable choice:
1. Manhattan yaw from wall normals (4*theta circular mean). Most interiors are rectilinear; aligning the
   frame to the walls turns plane fitting into 1-D peak finding and makes snapping trivial to defend.
2. Classify points by normal: floor / ceiling / wall facing +x, -x, +y, -y / other (furniture, clutter).
   Splitting walls by *facing direction* separates the two faces of a 10 cm interior wall for free.
3. Floor and ceiling: least-squares plane on class inliers (thousands of points -> mm-level standard
   error), ceiling height = ceiling plane minus floor plane at the room centroid.
4. Wall planes: histogram peaks of the coordinate per facing class, refined by iterative trimmed mean.
   Tall-support check rejects furniture faces (sofa backs, cabinets).
5. Rooms: floor occupancy grid, cut along every wall line (bridging door-sized gaps), connected
   components that the camera actually walked through. "A room is floor enclosed by walls."
6. Polygon: contour of the room mask, each edge snapped to the fitted wall plane facing into the room.
7. Openings: per wall, a u-v evidence grid of rays that *end on* the wall (solid) versus rays that
   *pass through* it (open). Doors touch the floor; windows have a sill. Edges refined on raw points.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient

from ..measure import Measurement

GRID = 0.02            # floor/wall evidence grid resolution (m)
CLASS_COS = 0.85       # |n . axis| needed to assign a point to a planar class (~32 deg)
FACES = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1)}
SYS_LEN = 0.005        # systematic floor on length uncertainty (LiDAR range bias), placeholder until conformal
Z95 = 1.645            # 90% two-sided


# ---------------------------------------------------------------------------------------------------
# fusion + classification
# ---------------------------------------------------------------------------------------------------
@dataclass
class Cloud:
    P: np.ndarray        # (N,3) world points (Manhattan-aligned)
    N: np.ndarray        # (N,3) normals
    cls: np.ndarray      # (N,) class label string codes
    cams: np.ndarray     # (F,3) camera centres (Manhattan-aligned)
    R: np.ndarray        # 4x4 Manhattan alignment applied on top of the poses/corrections
    rays: np.ndarray     # (M,4) sampled camera->hit segments in xy (cx, cy, px, py) for free-space carving


def rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def manhattan_yaw(N: np.ndarray, w: np.ndarray | None = None) -> float:
    """Dominant wall direction modulo 90 deg from horizontal normals: circular mean of 4*theta."""
    wall = np.abs(N[:, 2]) < 0.3
    th = np.arctan2(N[wall, 1], N[wall, 0])
    ww = None if w is None else w[wall]
    c, s = np.average(np.cos(4 * th), weights=ww), np.average(np.sin(4 * th), weights=ww)
    return float(np.arctan2(s, c) / 4)


def classify(N: np.ndarray) -> np.ndarray:
    cls = np.full(len(N), "other", dtype="<U5")
    cls[N[:, 2] > CLASS_COS] = "floor"
    cls[N[:, 2] < -CLASS_COS] = "ceil"
    cls[N[:, 0] > CLASS_COS] = "+x"
    cls[N[:, 0] < -CLASS_COS] = "-x"
    cls[N[:, 1] > CLASS_COS] = "+y"
    cls[N[:, 1] < -CLASS_COS] = "-y"
    return cls


def voxel_mean(P: np.ndarray, key_extra: np.ndarray, voxel: float):
    """Average points per (voxel, class) cell. Keeps classes separate so thin walls don't merge."""
    q = np.floor(P / voxel).astype(np.int64)
    k = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791) ^ (key_extra.astype(np.int64) * 2654435761)
    _, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
    out = np.zeros((len(cnt), 3))
    for d in range(3):
        out[:, d] = np.bincount(inv, P[:, d]) / cnt
    first = np.zeros(len(cnt), np.int64)
    first[inv[::-1]] = np.arange(len(inv))[::-1]
    return out, first


def fuse(cap, corrs: np.ndarray | None = None, frame_step: int = 2, voxel: float = 0.015,
         rays_per_frame: int = 300, seed: int = 0) -> Cloud:
    rng = np.random.default_rng(seed)
    Ps, Ns, cams, rays = [], [], [], []
    for i in range(0, len(cap.poses), frame_step):
        c = None if corrs is None else corrs[i]
        P, N, t = cap.world(i, c)
        Ps.append(P); Ns.append(N)
        if len(P):
            k = rng.choice(len(P), min(rays_per_frame, len(P)), replace=False)
            rays.append(np.c_[np.repeat(t[None, :2], len(k), 0), P[k, :2]])
    for i in range(len(cap.poses)):
        c = None if corrs is None else corrs[i]
        cams.append(cap.world(i, c)[2])
    P, N, cams, rays = np.concatenate(Ps), np.concatenate(Ns), np.asarray(cams), np.concatenate(rays)
    yaw = manhattan_yaw(N)
    R = rz(-yaw)
    R2 = R[:2, :2]
    P = P @ R[:3, :3].T
    N = N @ R[:3, :3].T
    cams = cams @ R[:3, :3].T
    rays = np.c_[rays[:, :2] @ R2.T, rays[:, 2:] @ R2.T]
    cls = classify(N)
    code = np.searchsorted(np.array(["+x", "+y", "-x", "-y", "ceil", "floor", "other"]), cls)
    Pv, first = voxel_mean(P, code, voxel)
    return Cloud(Pv, N[first], cls[first], cams, R, rays)


# ---------------------------------------------------------------------------------------------------
# horizontal planes
# ---------------------------------------------------------------------------------------------------
@dataclass
class HPlane:
    a: float
    b: float
    c: float             # z = a x + b y + c
    std: float
    n: int

    def z(self, x, y):
        return self.a * x + self.b * y + self.c


def pick_level(P: np.ndarray, side: str, min_frac: float = 0.03, min_extent_m2: float = 1.0) -> float:
    """Height of the floor (side='low') or ceiling (side='high') among horizontal-surface points.

    Not the most populated level: in furnished rooms bed and table tops often out-number visible floor.
    The floor is the LOWEST level, and the ceiling the HIGHEST, that has real mass (>= min_frac of the
    points within +-3 cm) and real horizontal extent (occupies >= min_extent_m2 at 10 cm cells)."""
    z = P[:, 2]
    edges = np.arange(z.min() - 0.02, z.max() + 0.03, 0.01)
    h, _ = np.histogram(z, edges)
    hs = gaussian_filter1d(h.astype(float), 1.0)
    order = range(len(hs)) if side == "low" else range(len(hs) - 1, -1, -1)
    for i in order:
        lvl = edges[i] + 0.005
        band = np.abs(z - lvl) < 0.03
        if band.sum() < max(100, min_frac * len(z)):
            continue
        if not ((i == 0 or hs[i] >= hs[i - 1]) and (i == len(hs) - 1 or hs[i] >= hs[i + 1])):
            continue
        cells = np.unique(np.floor(P[band, :2] / 0.1).astype(np.int64), axis=0)
        if len(cells) * 0.01 >= min_extent_m2:
            return float(lvl)
    return float(edges[np.argmax(hs)] + 0.005)


def fit_hplane(P: np.ndarray, z0: float | None = None, win: float = 0.04, iters: int = 4) -> HPlane:
    """Iteratively re-weighted least squares plane z = ax+by+c on points near z0."""
    if z0 is None:
        h, e = np.histogram(P[:, 2], bins=np.arange(P[:, 2].min(), P[:, 2].max() + 0.01, 0.01))
        z0 = e[np.argmax(gaussian_filter1d(h.astype(float), 1))] + 0.005
    sel = np.abs(P[:, 2] - z0) < win
    a = b = 0.0
    c = z0
    for _ in range(iters):
        Q = P[sel]
        if len(Q) < 10:
            break
        A = np.c_[Q[:, 0], Q[:, 1], np.ones(len(Q))]
        (a, b, c), *_ = np.linalg.lstsq(A, Q[:, 2], rcond=None)
        r = P[:, 2] - (a * P[:, 0] + b * P[:, 1] + c)
        s = max(np.std(r[sel]), 0.003)
        sel = np.abs(r) < 2.5 * s
    r = P[sel, 2] - (a * P[sel, 0] + b * P[sel, 1] + c)
    return HPlane(a, b, c, float(np.std(r)), int(sel.sum()))


# ---------------------------------------------------------------------------------------------------
# wall planes
# ---------------------------------------------------------------------------------------------------
@dataclass
class WallPlane:
    face: str            # '+x','-x','+y','-y'  (direction the wall surface faces, i.e. into its room)
    axis: int            # 0 -> plane x = offset ; 1 -> plane y = offset
    sign: int
    offset: float
    std: float           # residual std of inliers (m)
    n: int
    segments: list[tuple[float, float]] = field(default_factory=list)  # extents along the other axis

    @property
    def se(self) -> float:
        # standard error with a conservative effective sample size: neighbouring LiDAR points share pose
        # error, so they are far from independent; cap n_eff.
        return self.std / np.sqrt(min(self.n, 400))


def _peaks(h: np.ndarray, min_count: float, min_sep: int):
    order = np.argsort(h)[::-1]
    taken = []
    for i in order:
        if h[i] < min_count:
            break
        if all(abs(i - j) >= min_sep for j in taken) and (i == 0 or h[i] >= h[i - 1]) and (i == len(h) - 1 or h[i] >= h[i + 1]):
            taken.append(i)
    return sorted(taken)


def _segments(vals: np.ndarray, bridge: float, min_len: float = 0.3, min_count: int = 3):
    """Occupied extents of a 1-D coordinate set, bridging gaps up to `bridge`.

    Only doors leave a full-height gap in a wall (a window has wall below the sill), so `bridge` is a
    door width, not a window width. Bins need `min_count` points so a few misclassified corner points
    cannot extend a wall across a hallway."""
    if len(vals) == 0:
        return []
    b = np.floor(vals / GRID).astype(int)
    u, c = np.unique(b, return_counts=True)
    occ = u[c >= min_count]
    if len(occ) == 0:
        return []
    segs, start, prev = [], occ[0], occ[0]
    for o in occ[1:]:
        if (o - prev) * GRID > bridge:
            segs.append((start * GRID, (prev + 1) * GRID))
            start = o
        prev = o
    segs.append((start * GRID, (prev + 1) * GRID))
    return [s for s in segs if s[1] - s[0] >= min_len]


def fit_wall_planes(cloud: Cloud, floor: HPlane, ceil_z: float, bridge: float = 1.25) -> list[WallPlane]:
    planes = []
    zrel = cloud.P[:, 2] - floor.z(cloud.P[:, 0], cloud.P[:, 1])
    tall = min(1.9, ceil_z - 0.4)
    for face, (axis, sign) in FACES.items():
        m = (cloud.cls == face) & (zrel > 0.05) & (zrel < ceil_z - 0.05)
        Q = cloud.P[m]
        zq = zrel[m]
        if len(Q) < 100:
            continue
        x = Q[:, axis]
        edges = np.arange(x.min() - 0.02, x.max() + 0.03, 0.01)
        h, _ = np.histogram(x, edges)
        hs = gaussian_filter1d(h.astype(float), 1.5)
        for pk in _peaks(hs, min_count=max(60, 0.002 * len(Q)), min_sep=8):
            off = edges[pk] + 0.005
            win = 0.04
            for _ in range(4):
                sel = np.abs(x - off) < win
                if sel.sum() < 30:
                    break
                off = float(np.mean(x[sel]))
                win = max(2.5 * np.std(x[sel] - off), 0.012)
            sel = np.abs(x - off) < win
            if sel.sum() < 60:
                continue
            # tall support: real walls reach well above furniture height
            if np.percentile(zq[sel], 97) < tall:
                continue
            segs = _segments(Q[sel, 1 - axis], bridge)
            if not segs:
                continue
            planes.append(WallPlane(face, axis, sign, off, float(np.std(x[sel] - off)), int(sel.sum()), segs))
    return planes


# ---------------------------------------------------------------------------------------------------
# room segmentation
# ---------------------------------------------------------------------------------------------------
@dataclass
class Grid2:
    x0: float
    y0: float
    w: int
    h: int

    def ij(self, x, y):
        return ((np.asarray(y) - self.y0) / GRID).astype(int), ((np.asarray(x) - self.x0) / GRID).astype(int)

    def xy(self, i, j):
        return self.x0 + (np.asarray(j) + 0.5) * GRID, self.y0 + (np.asarray(i) + 0.5) * GRID


def segment_rooms(cloud: Cloud, floor: HPlane, walls: list[WallPlane], min_area: float = 1.0):
    """Floor occupancy cut by wall lines -> connected components that contain the camera path."""
    zrel = cloud.P[:, 2] - floor.z(cloud.P[:, 0], cloud.P[:, 1])
    fp = cloud.P[(cloud.cls == "floor") & (np.abs(zrel) < 0.04)]
    pad = 0.3
    g = Grid2(fp[:, 0].min() - pad, fp[:, 1].min() - pad,
              int((np.ptp(fp[:, 0]) + 2 * pad) / GRID) + 1, int((np.ptp(fp[:, 1]) + 2 * pad) / GRID) + 1)
    mask = np.zeros((g.h, g.w), np.uint8)
    i, j = g.ij(fp[:, 0], fp[:, 1])
    mask[i, j] = 1
    # Free space: every camera->hit segment crosses empty floor. This fills the ~1 m blind disc under a
    # handheld camera (it never looks straight down) and narrow hallways seen only at grazing angles.
    # Segments that leave the building through windows are cut off by the exterior wall lines below and
    # dropped because no camera position lies in them.
    free = np.zeros_like(mask)
    ci, cj = g.ij(cloud.rays[:, 0], cloud.rays[:, 1])
    pi_, pj = g.ij(cloud.rays[:, 2], cloud.rays[:, 3])
    for a, b, c, d in zip(cj, ci, pj, pi_):
        cv2.line(free, (int(a), int(b)), (int(c), int(d)), 1, 1)
    mask |= free
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # fill interior holes (furniture footprints) but not the outside
    ff = mask.copy()
    cv2.floodFill(ff, None, (0, 0), 2)
    mask[ff == 0] = 1
    cut = mask.copy()
    for wp in walls:
        for lo, hi in wp.segments:
            if wp.axis == 0:
                (i0, j0), (i1, j1) = g.ij(wp.offset, lo), g.ij(wp.offset, hi)
            else:
                (i0, j0), (i1, j1) = g.ij(lo, wp.offset), g.ij(hi, wp.offset)
            cv2.line(cut, (int(j0), int(i0)), (int(j1), int(i1)), 0, thickness=2)
    n, lab = cv2.connectedComponents(cut, connectivity=4)
    ci, cj = g.ij(cloud.cams[:, 0], cloud.cams[:, 1])
    inside = (ci >= 0) & (ci < g.h) & (cj >= 0) & (cj < g.w)
    visited = set(np.unique(lab[ci[inside], cj[inside]]).tolist()) - {0}
    rooms = []
    for k in range(1, n):
        m = (lab == k).astype(np.uint8)
        if m.sum() * GRID * GRID < min_area or k not in visited:
            continue
        rooms.append(m)
    return rooms, g


# ---------------------------------------------------------------------------------------------------
# polygons
# ---------------------------------------------------------------------------------------------------
@dataclass
class Edge:
    axis: int            # 0: vertical edge x = c ; 1: horizontal edge y = c
    c: float
    face: str            # wall facing into the room
    plane: WallPlane | None


def room_polygon(mask: np.ndarray, g: Grid2, walls: list[WallPlane]):
    """Contour -> rectilinear edges -> each edge snapped to the wall plane facing into the room."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    ap = cv2.approxPolyDP(cnt, 3, True)[:, 0, :].astype(float)
    xs, ys = g.xy(ap[:, 1], ap[:, 0])
    pts = np.c_[xs, ys]
    poly = Polygon(pts)
    if not poly.is_valid or poly.area <= 0:
        poly = poly.buffer(0)
    edges: list[Edge] = []
    n = len(pts)
    for k in range(n):
        p, q = pts[k], pts[(k + 1) % n]
        d = q - p
        L = np.hypot(*d)
        if L < 0.12:
            continue
        axis = 0 if abs(d[0]) < abs(d[1]) else 1    # vertical edge -> constant x
        if min(abs(d[0]), abs(d[1])) / L > 0.35:      # diagonal artefact of the staircase contour
            continue
        c = (p[axis] + q[axis]) / 2
        mid = (p + q) / 2
        probe = mid.copy()
        probe[axis] += 0.1
        sign = 1 if poly.contains(Point(probe)) else -1      # interior is on +axis side -> wall faces +axis
        face = ("+" if sign > 0 else "-") + "xy"[axis]
        lo, hi = sorted((p[1 - axis], q[1 - axis]))
        best, bd = None, 0.15
        for wp in walls:
            if wp.face != face:
                continue
            if not any(s0 < hi - 0.05 and s1 > lo + 0.05 for s0, s1 in wp.segments):
                continue
            if abs(wp.offset - c) < bd:
                best, bd = wp, abs(wp.offset - c)
        edges.append(Edge(axis, best.offset if best else c, face, best))
    # merge consecutive edges on the same axis (contour steps / noise) keeping the better supported one
    merged: list[Edge] = []
    for e in edges:
        if merged and merged[-1].axis == e.axis:
            prev = merged[-1]
            if prev.plane is None or (e.plane is not None and e.plane.n > prev.plane.n):
                merged[-1] = e
            continue
        merged.append(e)
    if len(merged) > 1 and merged[0].axis == merged[-1].axis:
        last = merged.pop()
        if merged[0].plane is None or (last.plane is not None and last.plane.n > merged[0].plane.n):
            merged[0] = last
    if len(merged) < 4:
        return None
    corners = []
    for k in range(len(merged)):
        a, b = merged[k - 1], merged[k]
        x = a.c if a.axis == 0 else b.c
        y = a.c if a.axis == 1 else b.c
        corners.append((x, y))
    # corner k is where edge k-1 meets edge k, so edge k runs corner k -> corner k+1
    walls_e = merged
    P = Polygon(corners)
    if not P.is_valid or P.area < 0.5:
        return None
    if not P.exterior.is_ccw:
        # reversed corner j is old corner n-1-j; reversed edge j (j -> j+1) is old edge n-2-j
        corners = corners[::-1]
        rev = walls_e[::-1]
        walls_e = rev[1:] + rev[:1]
    return corners, walls_e


# ---------------------------------------------------------------------------------------------------
# openings
# ---------------------------------------------------------------------------------------------------
@dataclass
class Face:
    room: int
    wall_idx: int
    axis: int            # plane coordinate axis (0: x=offset, 1: y=offset)
    sign: int            # normal direction (into room) along axis
    offset: float
    a: float             # extent along the other axis (world coords, a < b)
    b: float
    height: float


@dataclass
class OpeningEst:
    face: Face
    type: str
    u0: float            # world coordinate along wall
    u1: float
    v0: float            # height above floor
    v1: float
    conf: float
    width_se: float
    height_se: float


def opening_evidence(cap, frame_corr, faces: list[Face], room_polys: list[Polygon], floor: HPlane,
                     frame_step: int = 2, band: float = 0.04, beyond: float = 0.08):
    """Per face u-v grids: rays that end ON the wall (solid) vs rays that pass THROUGH it (open)."""
    grids = []
    for f in faces:
        nu = int((f.b - f.a) / GRID) + 1
        nv = int(f.height / GRID) + 1
        grids.append({"solid": np.zeros((nv, nu), np.int32), "thru": np.zeros((nv, nu), np.int32), "pts": []})
    for i in range(0, len(cap.poses), frame_step):
        P, Nw, c = cap.world(i, frame_corr(i))
        for fi, f in enumerate(faces):
            if not room_polys[f.room].buffer(-0.05).contains(Point(c[0], c[1])):
                continue
            G = grids[fi]
            k, u_ax = f.axis, 1 - f.axis
            d = (P[:, k] - f.offset) * f.sign       # signed distance, + inside the room
            # solid: points on the wall surface
            s = (np.abs(d) < band) & (Nw[:, k] * f.sign > 0.7)
            if s.any():
                u = P[s, u_ax]
                v = P[s, 2] - floor.z(P[s, 0], P[s, 1])
                ok = (u > f.a) & (u < f.b) & (v > 0) & (v < f.height)
                iu = ((u[ok] - f.a) / GRID).astype(int)
                iv = (v[ok] / GRID).astype(int)
                np.add.at(G["solid"], (iv, iu), 1)
                G["pts"].append(np.c_[u[ok], v[ok]])
            # through: points beyond the wall whose ray crosses the wall rectangle
            t_m = d < -beyond
            if t_m.any():
                Q = P[t_m]
                dc = (c[k] - f.offset) * f.sign
                if dc <= 0.05:
                    continue
                t = dc / (dc - d[t_m])           # fraction along camera->point where it crosses the plane
                X = c[None, :] + t[:, None] * (Q - c[None, :])
                u = X[:, u_ax]
                v = X[:, 2] - floor.z(X[:, 0], X[:, 1])
                ok = (u > f.a) & (u < f.b) & (v > 0) & (v < f.height)
                iu = ((u[ok] - f.a) / GRID).astype(int)
                iv = (v[ok] / GRID).astype(int)
                np.add.at(G["thru"], (iv, iu), 1)
    for G in grids:
        G["pts"] = np.concatenate(G["pts"]) if G["pts"] else np.zeros((0, 2))
    return grids


def _refine_edge(pts: np.ndarray, coarse: float, side: int, v0: float, v1: float, along: int = 0,
                 q: float = 97.0):
    """Sub-grid edge from raw solid points. side=-1: solid lies at smaller coordinate (gap above it);
    side=+1: solid at larger coordinate. Per 4 cm row, a high percentile of the solid coordinate
    approaching the gap; median over rows. Two passes: a wide window to find the edge, then a narrow
    one so the percentile's inward bias ((1-q) x window) is ~1 mm."""
    est_c, se = coarse, 0.02
    for win_out, win_in in ((0.15, 0.06), (0.05, 0.03)):
        if len(pts) == 0:
            break
        a, o = pts[:, along], pts[:, 1 - along]
        if side < 0:
            m = (a > est_c - win_out) & (a < est_c + win_in)
        else:
            m = (a < est_c + win_out) & (a > est_c - win_in)
        m &= (o > v0) & (o < v1)
        if m.sum() < 20:
            break
        rows = np.floor(o[m] / 0.04).astype(int)
        est = []
        for r in np.unique(rows):
            vals = a[m][rows == r]
            if len(vals) >= 5:
                est.append(np.percentile(vals, q) if side < 0 else np.percentile(vals, 100 - q))
        if len(est) < 3:
            break
        est = np.asarray(est)
        est_c, se = float(np.median(est)), float(1.2533 * np.std(est) / np.sqrt(len(est)))
    return est_c, se


def detect_openings(grids, faces: list[Face]) -> list[OpeningEst]:
    out = []
    for G, f in zip(grids, faces):
        S, T = G["solid"], G["thru"]
        openm = ((T >= 2) & (T > 2 * S)).astype(np.uint8)
        openm = cv2.morphologyEx(openm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        openm = cv2.morphologyEx(openm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(openm, connectivity=4)
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if w * GRID < 0.35 or h * GRID < 0.3 or area < 0.6 * w * h * 0.5:
                continue
            u0c, u1c = f.a + x * GRID, f.a + (x + w) * GRID
            v0c, v1c = y * GRID, (y + h) * GRID
            touches_floor = v0c < 0.15
            if touches_floor and v1c > f.height - 0.08:
                typ = "opening"
            elif touches_floor:
                typ = "door"
            else:
                typ = "window"
            mid0, mid1 = v0c + 0.1 * (v1c - v0c), v1c - 0.1 * (v1c - v0c)
            u0, s0 = _refine_edge(G["pts"], u0c, -1, mid0, mid1, along=0)
            u1, s1 = _refine_edge(G["pts"], u1c, +1, mid0, mid1, along=0)
            # vertical edges: head (solid above) and sill (solid below)
            cu0, cu1 = u0 + 0.1 * (u1 - u0), u1 - 0.1 * (u1 - u0)
            pts_vu = G["pts"][:, ::-1] if len(G["pts"]) else G["pts"]
            if typ == "opening":
                v1, s3 = f.height, 0.0
            else:
                v1, s3 = _refine_edge(pts_vu, v1c, +1, cu0, cu1, along=0)
            if typ == "window":
                v0, s2 = _refine_edge(pts_vu, v0c, -1, cu0, cu1, along=0)
            else:
                v0, s2 = 0.0, 0.0
            frac = (T[y:y + h, x:x + w].sum() + 1) / (T[y:y + h, x:x + w].sum() + S[y:y + h, x:x + w].sum() + 1)
            out.append(OpeningEst(f, typ, u0, u1, v0, v1, float(frac),
                                  float(np.hypot(s0, s1)), float(np.hypot(s2, s3))))
    return out


# ---------------------------------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------------------------------
def meas_len(value: float, se: float, method: str = "fitstat:v0") -> Measurement:
    return Measurement.from_abs(value, Z95 * np.hypot(se, SYS_LEN), method=method)


def ceiling_height(cloud: Cloud, floor: HPlane, poly: Polygon):
    inner = poly.buffer(-0.15)
    if inner.is_empty:
        inner = poly
    m = cloud.cls == "ceil"
    Q = cloud.P[m]
    from shapely import contains_xy
    Q = Q[contains_xy(inner, Q[:, 0], Q[:, 1])]
    if len(Q) < 50:
        return None
    zr = Q[:, 2] - floor.z(Q[:, 0], Q[:, 1])
    Qr = np.c_[Q[:, :2], zr]
    cp = fit_hplane(Qr, z0=pick_level(Qr, "high", min_extent_m2=min(1.0, 0.3 * poly.area)))
    cx, cy = poly.centroid.x, poly.centroid.y
    h = cp.z(cx, cy)
    se = np.hypot(cp.std / np.sqrt(min(cp.n, 400)), floor.std / np.sqrt(min(floor.n, 400)))
    return h, se, cp.n
