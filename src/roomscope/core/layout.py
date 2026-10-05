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
from scipy.ndimage import distance_transform_edt as ndimage_edt
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient

from ..measure import Measurement

GRID = 0.02            # floor/wall evidence grid resolution (m)
CLASS_COS = 0.85       # |n . axis| needed to assign a point to a planar class (~32 deg)
FACES = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1)}
SNAP_DIST = 0.30      # floor-boundary edge -> wall plane snapping radius (m)
LAYOUT_CONFIG = __import__("pathlib").Path(__file__).resolve().parents[3] / "config" / "layout.yaml"
SYS_LEN = 0.005        # systematic floor on length uncertainty (LiDAR range bias), placeholder until conformal
Z95 = 1.645            # 90% two-sided


def layout_config() -> dict:
    """Physical priors for layout (config/layout.yaml)."""
    import yaml
    return yaml.safe_load(LAYOUT_CONFIG.read_text())


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
    rays: np.ndarray     # (M,6) sampled camera->hit segments (cx, cy, cz, px, py, pz): free space + see-over tests


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
         rays_per_frame: int = 600, seed: int = 0) -> Cloud:
    rng = np.random.default_rng(seed)
    Ps, Ns, cams, rays = [], [], [], []
    for i in range(0, len(cap.poses), frame_step):
        c = None if corrs is None else corrs[i]
        P, N, t = cap.world(i, c)
        Ps.append(P); Ns.append(N)
        if len(P):
            k = rng.choice(len(P), min(rays_per_frame, len(P)), replace=False)
            rays.append(np.c_[np.repeat(t[None, :], len(k), 0), P[k]])
    for i in range(len(cap.poses)):
        c = None if corrs is None else corrs[i]
        cams.append(cap.world(i, c)[2])
    P, N, cams, rays = np.concatenate(Ps), np.concatenate(Ns), np.asarray(cams), np.concatenate(rays)
    yaw = manhattan_yaw(N)
    R = rz(-yaw)
    P = P @ R[:3, :3].T
    N = N @ R[:3, :3].T
    cams = cams @ R[:3, :3].T
    rays = np.c_[rays[:, :3] @ R[:3, :3].T, rays[:, 3:] @ R[:3, :3].T]
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


def pick_level(P: np.ndarray, side: str, min_frac: float = 0.03, min_extent_m2: float = 1.0,
                cluster: float = 0.15) -> float:
    """Height of the floor (side='low') or ceiling (side='high') among horizontal-surface points.

    Not simply the most populated level: in furnished rooms bed and table tops often out-number visible
    floor. So first find the LOWEST level (floor) or HIGHEST level (ceiling) with real mass (>= min_frac
    of the points within +-3 cm) and real horizontal extent (>= min_extent_m2 at 10 cm cells). Then,
    among qualifying levels within `cluster` of it, take the best supported one. Odometry height drift
    leaves a thin phantom copy of the floor or ceiling a few cm off: on 42444946 without drift
    correction, a ceiling layer 8 cm above the real one held 15% of the points, and "highest" picked it
    (+6.6 cm). Furniture tops sit >= 40 cm from the floor, so the 15 cm cluster does not reach them."""
    z = P[:, 2]
    edges = np.arange(z.min() - 0.02, z.max() + 0.03, 0.01)
    h, _ = np.histogram(z, edges)
    hs = gaussian_filter1d(h.astype(float), 1.0)
    order = range(len(hs)) if side == "low" else range(len(hs) - 1, -1, -1)
    cands = []                                  # (level, band count), from the extreme inward
    for i in order:
        lvl = edges[i] + 0.005
        if cands and abs(lvl - cands[0][0]) > cluster:
            break
        band = np.abs(z - lvl) < 0.03
        if band.sum() < max(100, min_frac * len(z)):
            continue
        if not ((i == 0 or hs[i] >= hs[i - 1]) and (i == len(hs) - 1 or hs[i] >= hs[i + 1])):
            continue
        cells = np.unique(np.floor(P[band, :2] / 0.1).astype(np.int64), axis=0)
        if len(cells) * 0.01 >= min_extent_m2:
            cands.append((float(lvl), int(band.sum())))
    if cands:
        return max(cands, key=lambda c: c[1])[0]
    return float(edges[np.argmax(hs)] + 0.005)


def fit_hplane(P: np.ndarray, z0: float | None = None, win: float = 0.04, iters: int = 4,
               tilt: bool = False) -> HPlane:
    """Horizontal level (or tilted plane z = ax+by+c if tilt=True) from points near z0, iteratively trimmed.

    Level by default: the frame is gravity-aligned (ARKit), real floors and ceilings are level to ~1 cm,
    and fitting a tilt to a ceiling seen only in patches, then evaluating it at the room centre,
    extrapolates noise (it gave 2.16 m for a 2.44 m ceiling on scene 41069042)."""
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
        if tilt:
            A = np.c_[Q[:, 0], Q[:, 1], np.ones(len(Q))]
            (a, b, c), *_ = np.linalg.lstsq(A, Q[:, 2], rcond=None)
        else:
            c = float(np.mean(Q[:, 2]))
        r = P[:, 2] - (a * P[:, 0] + b * P[:, 1] + c)
        s_ = max(np.std(r[sel]), 0.003)
        sel = np.abs(r) < min(2.5 * s_, win)
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
    top: float = 0.0     # 97th percentile height of its points above the floor (how high it was observed)
    kind: str = "wall"   # "wall" separates rooms; "builtin" (tall but seen over: wardrobe, closet) only bounds the polygon

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


def seen_over_fraction(cloud: Cloud, floor: HPlane, axis: int, sign: int, offset: float,
                       segs: list[tuple[float, float]], face_u: np.ndarray, face_v: np.ndarray, ceil_z: float,
                       beyond: float = 0.25, bin_m: float = 0.10) -> float:
    """Fraction of a candidate plane's length over which we can SEE OVER it.

    A wall hides what is behind it at every height, up to the ceiling; furniture (a dresser front, a bed
    end, a sofa back) has empty space above its top. So count rays whose camera is on the facing side,
    whose hit lies well beyond the plane, and which cross the plane ABOVE the face's LOCAL observed top
    (per 10 cm bin: a bed side is 0.6 m along most of its length even if its headboard reaches 1.1 m). If
    that happens along most of the length, it is furniture. Doors and windows only open part of a wall,
    so a real wall with openings keeps a low fraction. This works when the scan never reached the
    ceiling, where a fixed "must be tall" test rejects real walls (scene 41069042: walls seen only to
    1.3 m because the device was held low)."""
    R = cloud.rays
    c, p = R[:, :3], R[:, 3:]
    dc = (c[:, axis] - offset) * sign
    dp = (p[:, axis] - offset) * sign
    m = (dc > 0.2) & (dp < -beyond)
    if not m.any():
        return 0.0
    t = dc[m] / (dc[m] - dp[m])
    X = c[m] + t[:, None] * (p[m] - c[m])
    v = X[:, 2] - floor.z(X[:, 0], X[:, 1])
    u = X[:, 1 - axis]
    total, covered = 0, 0
    for lo, hi in segs:
        nb = max(1, int((hi - lo) / bin_m))
        fb = ((face_u - lo) / bin_m).astype(int)
        okf = (fb >= 0) & (fb < nb)
        local_top = np.zeros(nb)
        np.maximum.at(local_top, fb[okf], face_v[okf])     # bins without face points: top 0 (open)
        rb = ((u - lo) / bin_m).astype(int)
        okr = (rb >= 0) & (rb < nb)
        over = okr.copy()
        over[okr] = (v[okr] > local_top[rb[okr]] + 0.05) & (v[okr] < ceil_z - 0.05)
        cnt = np.bincount(rb[over], minlength=nb)
        total += nb
        covered += int((cnt >= 2).sum())
    return covered / max(total, 1)


def fit_wall_planes(cloud: Cloud, floor: HPlane, ceil_z: float, bridge: float = 1.25,
                    min_top: float = 1.0, max_seen_over: float = 0.5) -> list[WallPlane]:
    planes = []
    zrel = cloud.P[:, 2] - floor.z(cloud.P[:, 0], cloud.P[:, 1])
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
            top = float(np.percentile(zq[sel], 97))
            if top < min_top:                     # beds, dressers, tables: too low to be a wall
                continue
            segs = _segments(Q[sel, 1 - axis], bridge)
            if not segs:
                continue
            kind = "wall"
            if seen_over_fraction(cloud, floor, axis, sign, off, segs, Q[sel, 1 - axis], zq[sel], ceil_z) > max_seen_over:
                # we can see over it: furniture. Tall built-ins (wardrobes, closets) still bound the floor plan.
                if top < 1.8:
                    continue
                kind = "builtin"
            planes.append(WallPlane(face, axis, sign, off, float(np.std(x[sel] - off)), int(sel.sum()), segs, top, kind))
    return drop_layered(planes, ceil_z)


def _overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(h1, h2) - max(l1, l2)) for l1, h1 in a for l2, h2 in b)


def drop_layered(planes: list[WallPlane], ceil_z: float, max_gap: float = 0.6, cover: float = 0.6,
                 max_std: float = 0.045, top_margin: float = 0.6, min_support: float = 0.3) -> list[WallPlane]:
    """Remove surfaces standing in front of a wall: a wardrobe face, a curtain, a radiator panel.

    A plane is dropped when another plane facing the same way lies BEHIND it (further from the room,
    within max_gap), covers >= `cover` of its extent, and is wall-like: observed to within top_margin
    of the ceiling, and flat (residual std <= max_std; curtain folds give 7-9 cm). Found on 42444946:
    - a 1.95 m wardrobe 17 cm in front of the wall;
    - a curtain 17 cm in front of a window wall.
    Both had been taken as the wall. A wall jog or alcove is unaffected: there the rear plane runs
    beside the front one rather than behind it, so coverage is low.
    top_margin is 0.6 m because a window wall is often observed only to the curtain rail (2.65 m under
    a 3.06 m ceiling on 42444949). The rear plane also needs >= min_support of the front plane's points,
    so a faint spurious plane cannot remove a strong real wall (42444949, left wall)."""
    def wall_like(p: WallPlane) -> bool:
        return p.top >= ceil_z - top_margin and p.std <= max_std

    drop = set()
    for i, p in enumerate(planes):
        ext = sum(h - l for l, h in p.segments)
        for j, q in enumerate(planes):
            if i == j or q.face != p.face or not wall_like(q) or q.n < min_support * p.n:
                continue
            behind = (p.offset - q.offset) * p.sign      # face '+x': room at larger x, so behind = smaller x
            if 0.03 < behind <= max_gap and _overlap(p.segments, q.segments) >= cover * ext:
                drop.add(i)
                break
    return [p for i, p in enumerate(planes) if i not in drop]


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


@dataclass
class Cut:
    """An axis-aligned wall line used to separate rooms. axis 0: x = c for y in [lo, hi]; axis 1: y = c."""
    axis: int
    c: float
    lo: float
    hi: float
    observed: bool       # False for an extension that closes an unobserved stretch


def _split_by_path(cut: "Cut", cams_xy: np.ndarray | None, clear: float = 0.4,
                   min_len: float = 0.3) -> list["Cut"]:
    """Pieces of `cut` that stay more than `clear` from every point where the camera path crosses it."""
    if cams_xy is None or len(cams_xy) < 2:
        return [cut]
    k, o = cut.axis, 1 - cut.axis
    a, b = cams_xy[:-1], cams_xy[1:]
    sa, sb = a[:, k] - cut.c, b[:, k] - cut.c
    cr = sa * sb < 0
    if not cr.any():
        return [cut]
    t = sa[cr] / (sa[cr] - sb[cr])
    along = a[cr, o] + t * (b[cr, o] - a[cr, o])
    along = np.sort(along[(along > cut.lo - clear) & (along < cut.hi + clear)])
    if not len(along):
        return [cut]
    pieces, start = [], cut.lo
    for x in along:
        if x - clear - start >= min_len:
            pieces.append(Cut(cut.axis, cut.c, start, x - clear, cut.observed))
        start = max(start, x + clear)
    if cut.hi - start >= min_len:
        pieces.append(Cut(cut.axis, cut.c, start, cut.hi, cut.observed))
    return pieces


def _crosses_path(cut: "Cut", cams_xy: np.ndarray) -> bool:
    """Does the camera trajectory cross this axis-aligned segment? Nobody walks through a wall."""
    if cams_xy is None or len(cams_xy) < 2:
        return False
    k, o = cut.axis, 1 - cut.axis
    a, b = cams_xy[:-1], cams_xy[1:]
    side_a, side_b = a[:, k] - cut.c, b[:, k] - cut.c
    cross = side_a * side_b < 0
    if not cross.any():
        return False
    t = side_a[cross] / (side_a[cross] - side_b[cross])
    along = a[cross, o] + t * (b[cross, o] - a[cross, o])
    return bool(np.any((along > cut.lo) & (along < cut.hi)))


def wall_cuts(walls: list[WallPlane], corner_tol: float = 0.15, max_ext: float = 3.0,
              cams_xy: np.ndarray | None = None) -> list[Cut]:
    """Observed wall segments, plus extensions of their OPEN ends until they meet another wall line.

    Why: a room is closed by walls, not by how far the floor happened to be seen. Where a wall stops
    without meeting a perpendicular wall (it ran out of scan, or reached a wide opening), we continue
    its line until it meets another wall or another such extension, so free space seen through the gap
    cannot join the room. Ends that already meet a perpendicular wall (corners) are not extended, so an
    L-shaped room is not split along the line of its inner corner."""
    cuts = [Cut(w.axis, w.offset, lo, hi, True) for w in walls if w.kind == "wall" for lo, hi in w.segments]
    # nobody walks through a wall: drop the stretch of an observed segment the camera path crosses
    # (segments are bridged over 1.25 m gaps, so a door leaf plus stray points can form a fake wall
    # across a room; on 42444946 that split one room in two)
    cuts = [p for c in cuts for p in _split_by_path(c, cams_xy)]

    def is_corner(cut: Cut, u: float) -> bool:
        for o in cuts:
            if o is cut:
                continue
            if o.axis != cut.axis and abs(o.c - u) < corner_tol and o.lo - corner_tol <= cut.c <= o.hi + corner_tol:
                return True
            if o.axis == cut.axis and abs(o.c - cut.c) < 0.05 and o.lo - corner_tol <= u <= o.hi + corner_tol \
                    and not (cut.lo <= o.lo and o.hi <= cut.hi):
                return True     # collinear continuation already covers this end
        return False

    rays = []                   # (cut, end u, direction)
    for cut in cuts:
        for u, d in ((cut.lo, -1), (cut.hi, +1)):
            if not is_corner(cut, u):
                rays.append((cut, u, d))

    exts = []
    for cut, u, d in rays:
        best = max_ext
        for o in cuts:          # perpendicular observed walls ahead
            if o.axis != cut.axis and o.lo - 0.05 <= cut.c <= o.hi + 0.05:
                t = (o.c - u) * d
                if 0.02 < t < best:
                    best = t
            elif o is not cut and o.axis == cut.axis and abs(o.c - cut.c) < 0.05:
                t = ((o.lo if d > 0 else o.hi) - u) * d     # collinear segment ahead: join them
                if 0.02 < t < best:
                    best = t
        for cut2, u2, d2 in rays:   # another open end coming the perpendicular way
            if cut2.axis == cut.axis:
                continue
            t1 = (cut2.c - u) * d   # along our line to their line
            t2 = (cut.c - u2) * d2  # along their line to ours
            if 0.02 < t1 < best and 0.0 <= t2 <= max_ext:
                best = t1
        if best < max_ext:
            lo, hi = sorted((u, u + d * best))
            ext = Cut(cut.axis, cut.c, lo, hi, False)
            if not _crosses_path(ext, cams_xy):    # the camera walked through it: not a wall
                exts.append(ext)
    return cuts + exts


def _clip_rays(rays2: np.ndarray, cuts: list[Cut]) -> np.ndarray:
    """Shorten each 2-D camera->hit segment at the first cut it crosses (free space stops at walls)."""
    c, p = rays2[:, :2], rays2[:, 2:]
    d = p - c
    s_min = np.ones(len(c))
    for cut in cuts:
        k, o = cut.axis, 1 - cut.axis
        den = d[:, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (cut.c - c[:, k]) / den
        along = c[:, o] + s * d[:, o]
        hit = (s > 1e-3) & (s < s_min) & (along >= cut.lo) & (along <= cut.hi) & (np.abs(den) > 1e-9)
        s_min[hit] = s[hit]
    return np.c_[c, c + d * s_min[:, None]]


def segment_rooms(cloud: Cloud, floor: HPlane, walls: list[WallPlane], min_area: float = 1.0,
                  single_room: bool = False):
    """Floor evidence enclosed by wall lines -> connected components that contain the camera path.

    single_room (photo tier: one folder = one room by protocol): 2-8 photos see the floor only in
    patches, because furniture hides it between views (42444946: four disconnected patches 1.5-2 m apart
    inside walls 4.2 x 4.2 m apart; the camera-path rule kept one 2.4 m2 patch). Then the room is the
    convex hull of the floor patches and wall points, snapped to the wall planes by room_polygon."""
    cuts = wall_cuts(walls, cams_xy=cloud.cams[:, :2])
    # built-in fronts (wardrobes, closets) bound floor and stop free space, but are never extended, so a
    # free-standing tall cabinet cannot split a room: the room stays connected around its ends
    cuts = cuts + [Cut(w.axis, w.offset, lo, hi, True) for w in walls if w.kind == "builtin" for lo, hi in w.segments]
    zrel = cloud.P[:, 2] - floor.z(cloud.P[:, 0], cloud.P[:, 1])
    fp = cloud.P[(cloud.cls == "floor") & (np.abs(zrel) < 0.04)]
    pad = 0.3
    ext = fp[:, :2]
    if single_room:                 # the grid must also cover walls standing beyond the visible floor
        wl0 = np.isin(cloud.cls, ["+x", "-x", "+y", "-y"]) & (zrel > 0.1) & (zrel < 2.0)
        ext = np.concatenate([ext, cloud.P[wl0, :2]])
    g = Grid2(ext[:, 0].min() - pad, ext[:, 1].min() - pad,
              int((np.ptp(ext[:, 0]) + 2 * pad) / GRID) + 1, int((np.ptp(ext[:, 1]) + 2 * pad) / GRID) + 1)
    mask = np.zeros((g.h, g.w), np.uint8)
    i, j = g.ij(fp[:, 0], fp[:, 1])
    mask[i, j] = 1
    # Free space: a camera->hit segment crosses empty floor, but only up to the first wall line it meets.
    # This fills the ~1 m blind disc under a handheld camera without letting space seen THROUGH an opening
    # (another room, an unscanned area) count as this room.
    free = np.zeros_like(mask)
    # only rays that end no higher than the camera: they cannot pass OVER anything taller than the camera
    # (a wardrobe), so their footprint is genuinely free floor; rays over a bed are fine (bed is in the room)
    low = cloud.rays[:, 5] <= cloud.rays[:, 2] + 0.05
    r2 = _clip_rays(cloud.rays[low][:, [0, 1, 3, 4]], cuts)
    ci, cj = g.ij(r2[:, 0], r2[:, 1])
    pi_, pj = g.ij(r2[:, 2], r2[:, 3])
    for a, b, c, d in zip(cj, ci, pj, pi_):
        cv2.line(free, (int(a), int(b)), (int(c), int(d)), 1, 1)
    mask |= free
    # The strongest free-space evidence is the camera itself: the floor around where the phone was held is
    # empty, yet no ray from a camera that looks outwards ever crosses it. Without this, an operator who
    # walks a loop in the middle of a room leaves an unobserved hole there (c00a170fe1: 0.6 x 1.2 m, open to
    # the room's edge, so it was not filled and the room outline could not be built).
    rcfg = layout_config().get("rooms", {})
    rad = int(round(rcfg.get("camera_free_radius_m", 0.0) / GRID))
    if rad > 0:
        ci, cj = g.ij(cloud.cams[:, 0], cloud.cams[:, 1])
        for a, b in zip(cj, ci):
            cv2.circle(mask, (int(a), int(b)), rad, 1, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cut = mask.copy()
    for ct in cuts:
        if ct.axis == 0:
            (i0, j0), (i1, j1) = g.ij(ct.c, ct.lo), g.ij(ct.c, ct.hi)
        else:
            (i0, j0), (i1, j1) = g.ij(ct.lo, ct.c), g.ij(ct.hi, ct.c)
        cv2.line(cut, (int(j0), int(i0)), (int(j1), int(i1)), 0, thickness=2)
    if single_room:
        # convex hull of the floor patches and the low wall points (walls bound the room even where the
        # floor in front of them is hidden); room_polygon then snaps its edges to the wall planes.
        # Known limit: an L-shaped room is filled to its hull.
        wl = np.isin(cloud.cls, ["+x", "-x", "+y", "-y"]) & (zrel > 0.1) & (zrel < 2.0)
        pts = np.concatenate([fp[:, :2], cloud.P[wl, :2]])
        pts = pts[(pts[:, 0] >= g.x0) & (pts[:, 1] >= g.y0)]
        ii, jj = g.ij(pts[:, 0], pts[:, 1])
        okk = (ii >= 0) & (ii < g.h) & (jj >= 0) & (jj < g.w)
        if okk.sum() >= 3:
            hull = cv2.convexHull(np.stack([jj[okk], ii[okk]], 1).astype(np.int32))
            m = np.zeros_like(mask)
            cv2.fillConvexPoly(m, hull, 1)
            return [m], g, cuts, [float(max(0, m.sum() - (mask & m).sum()) * GRID * GRID)]
    if rcfg.get("method") == "scp":
        lab = _scp_labels(cloud, zrel, mask, g, rcfg)
        n = int(lab.max()) + 1
        min_area = rcfg.get("min_room_area_m2", min_area)
    else:
        n, lab = cv2.connectedComponents(cut, connectivity=4)
    ci, cj = g.ij(cloud.cams[:, 0], cloud.cams[:, 1])
    inside = (ci >= 0) & (ci < g.h) & (cj >= 0) & (cj < g.w)
    visited = set(np.unique(lab[ci[inside], cj[inside]]).tolist()) - {0}
    # barrier raster for hole filling: wall cuts AND built-in faces (a closet front bounds the room's
    # floor even though it does not separate rooms)
    barrier = np.zeros_like(mask)
    lines = [(ct.axis, ct.c, ct.lo, ct.hi) for ct in cuts]
    for axis, c0, lo, hi in lines:
        if axis == 0:
            (i0, j0), (i1, j1) = g.ij(c0, lo), g.ij(c0, hi)
        else:
            (i0, j0), (i1, j1) = g.ij(lo, c0), g.ij(hi, c0)
        cv2.line(barrier, (int(j0), int(i0)), (int(j1), int(i1)), 1, thickness=3)
    k30 = np.ones((int(0.3 / GRID), int(0.3 / GRID)), np.uint8)
    rooms, inferred = [], []
    segment_rooms.not_entered = []
    for k in range(1, n):
        m = (lab == k).astype(np.uint8)
        if m.sum() * GRID * GRID < min_area or k not in visited:
            continue
        # Fill floor never seen directly (furniture footprints, a corner the camera skirted) when it is
        # enclosed by this room plus walls. Drop fill thinner than 30 cm: that is the gap between two
        # faces of a wall or a furniture slot, not floor.
        ff = np.pad(((m | barrier) > 0).astype(np.uint8), 1)
        cv2.floodFill(ff, None, (0, 0), 2)
        # never another room's floor: with a partition that does not follow wall cuts (SCP), the
        # neighbouring rooms are enclosed by the same outer walls
        holes = ((ff[1:-1, 1:-1] == 0) & (m == 0) & (barrier == 0) & ((lab == 0) | (lab == k))).astype(np.uint8)
        holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, k30)
        # only holes that touch the room directly; space sealed off behind a barrier (a closet behind its
        # front, the cavity between two wall faces) is not this room's floor
        nh, hl = cv2.connectedComponents(holes, connectivity=4)
        touch = cv2.dilate(m * (1 - barrier), np.ones((3, 3), np.uint8)) > 0
        # real contact (>= 30 cm of shared boundary), not a pixel leak where two barrier lines meet
        cnt = np.bincount(hl[touch & (hl > 0)], minlength=nh)
        keep = {k for k in range(1, nh) if cnt[k] >= int(0.3 / GRID)}
        holes = np.isin(hl, list(keep)).astype(np.uint8) if keep else np.zeros_like(holes)
        seen = m
        m = (m | holes).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        if not _entered(m, g, cloud.cams[:, :2], rcfg, (lab > 0) & (lab != k)):   # on the filled room
            segment_rooms.not_entered.append(float(m.sum() * GRID * GRID))
            continue
        rooms.append(m)
        inferred.append(seen)
    # Each room fills unobserved floor on its own, so two rooms can claim the same unseen patch
    # (1a8384c3f6: 1.75 m2, drawn twice). A contested cell goes to the room whose own floor
    # evidence is nearest.
    if len(rooms) > 1:
        claims = np.sum(rooms, axis=0)
        if (claims > 1).any():
            dist = np.stack([ndimage_edt(sn == 0) for sn in inferred])
            owner = np.argmin(dist, axis=0)
            for k in range(len(rooms)):
                rooms[k] = (rooms[k] & ((claims <= 1) | (owner == k))).astype(np.uint8)
    inferred = [float(max(0, int(m.sum()) - int((m & sn).sum())) * GRID * GRID) for m, sn in zip(rooms, inferred)]
    return rooms, g, cuts, inferred


def _entered(m: np.ndarray, g: Grid2, cams_xy: np.ndarray, rcfg: dict, others: np.ndarray | None = None) -> bool:
    """Did the camera walk into this region? A room is kept only if the scanner path entered it
    (Turner & Zakhor 2014): free space seen through a doorway from outside is not a scanned room.
    Entered = at least entered_path_m of the camera path (interpolated, so sparse video keyframes
    count the same as dense LiDAR frames) lies inside the region and more than entered_depth_m from
    any OTHER region, i.e. past the doorway. Depth is not measured from walls or furniture: in a
    furnished room the camera walks narrow strips between them."""
    depth, need = rcfg.get("entered_depth_m", 0.3), rcfg.get("entered_path_m", 0.5)
    core = m.copy()
    if others is not None and others.any():
        far = ndimage_edt(others == 0) * GRID > depth
        core = (m > 0) & far
    if len(cams_xy) < 2:
        return bool(len(cams_xy)) and bool(core[tuple(np.clip(g.ij(*cams_xy[0]), 0, [g.h - 1, g.w - 1]))])
    seg = np.linalg.norm(np.diff(cams_xy, axis=0), axis=1)
    step = 0.05
    pts = np.concatenate([cams_xy[i] + np.outer(np.arange(0, 1, step / max(L_, step)), cams_xy[i + 1] - cams_xy[i])
                          for i, L_ in enumerate(seg)] + [cams_xy[-1:]])
    lens = np.concatenate([np.full(max(1, len(np.arange(0, 1, step / max(L_, step)))), min(L_, step))
                           for L_ in seg] + [[0.0]])
    i, j = g.ij(pts[:, 0], pts[:, 1])
    ok = (i >= 0) & (i < g.h) & (j >= 0) & (j < g.w)
    return float(lens[ok][core[i[ok], j[ok]] > 0].sum()) >= need


def _scp_labels(cloud: Cloud, zrel: np.ndarray, mask: np.ndarray, g: Grid2, rcfg: dict) -> np.ndarray:
    """Room labels on the floor grid by structural-clearance persistence (core/rooms_scp.py).

    Obstacles are wall-face points in a height band (furniture is mostly lower; scans held low still
    see walls there); free = observed floor + free space from camera rays, minus obstacles. The
    partition runs on a coarser grid (it is scale-free) and is mapped back to the floor grid."""
    from . import rooms_scp
    sc = rcfg["scp"]
    lo, hi = sc["obstacle_band_m"]
    wl = np.isin(cloud.cls, ["+x", "-x", "+y", "-y"]) & (zrel >= lo) & (zrel <= hi)
    obst = np.zeros_like(mask)
    i, j = g.ij(cloud.P[wl, 0], cloud.P[wl, 1])
    ok = (i >= 0) & (i < g.h) & (j >= 0) & (j < g.w)
    obst[i[ok], j[ok]] = 1
    obst = cv2.dilate(obst, np.ones((3, 3), np.uint8))
    free = (mask > 0) & (obst == 0)
    f = max(1, int(round(sc["cell_m"] / GRID)))
    H, W = (g.h + f - 1) // f * f, (g.w + f - 1) // f * f
    pad = np.zeros((H, W), np.float32)
    pad[:g.h, :g.w] = free
    coarse = pad.reshape(H // f, f, W // f, f).mean((1, 3)) >= 0.5
    part = rooms_scp.partition(coarse, GRID * f, (float(np.log(sc["ratio_band"][0])), float(np.log(sc["ratio_band"][1]))),
                               sc["r_min_m"], sc["r_floor_m"])
    lab = np.repeat(np.repeat(part.labels, f, 0), f, 1)[:g.h, :g.w]
    lab = np.where(mask > 0, lab, 0).astype(np.int32)
    segment_rooms.scp = part           # diagnostics: tau, seeds, margins
    return lab


# ---------------------------------------------------------------------------------------------------
# polygons
# ---------------------------------------------------------------------------------------------------
@dataclass
class Edge:
    axis: int            # 0: vertical edge x = c ; 1: horizontal edge y = c
    c: float
    face: str            # wall facing into the room
    plane: WallPlane | None
    shared: bool = False # open boundary with another room (no wall): position from the room partition


def room_polygon(mask: np.ndarray, g: Grid2, walls: list[WallPlane], others: np.ndarray | None = None):
    """Contour -> rectilinear edges -> each edge snapped to the wall plane facing into the room.

    others: the other rooms' floor masks. An edge whose outside is mostly another room's floor is an open
    boundary (doorway, open junction) shared with that room: it is never snapped to a wall plane, so both
    rooms keep the same partition line instead of each snapping to a different nearby wall face
    (c00a170fe1: corridor and room snapped 15 cm apart, overlapping by 0.25 m2)."""
    nb = cv2.dilate(others.astype(np.uint8), np.ones((3, 3), np.uint8)) if others is not None else None
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
        if nb is not None:
            t = np.linspace(0, 1, max(2, int(L / GRID)))
            probe_pts = p[None, :] + t[:, None] * d[None, :]
            probe_pts[:, axis] -= sign * 1.5 * GRID          # just outside the room
            pi_, pj = g.ij(probe_pts[:, 0], probe_pts[:, 1])
            ok = (pi_ >= 0) & (pi_ < g.h) & (pj >= 0) & (pj < g.w)
            if ok.any() and nb[pi_[ok], pj[ok]].mean() >= 0.5:
                edges.append(Edge(axis, c, face, None, shared=True))
                continue
        # Observed walls beat floor extent: snap to the facing wall plane within SNAP_DIST of the floor
        # boundary. Prefer planes whose observed extent overlaps the edge; a plane whose line merely
        # continues past the edge (an extended wall) is allowed at a penalty.
        best, bs = None, np.inf
        for wp in walls:
            if wp.face != face or abs(wp.offset - c) > SNAP_DIST:
                continue
            gap = min(max(0.0, s0 - hi, lo - s1) for s0, s1 in wp.segments)
            if gap > 3.0:
                continue
            # a face never seen above 1.8 m may be furniture standing against the wall: prefer a taller
            # plane behind it when both are within snapping range (scene 41069042: dresser front vs wall)
            score = abs(wp.offset - c) + 0.1 * gap + (0.15 if wp.top < 1.8 else 0.0)
            if score < bs:
                best, bs = wp, score
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
    def corners_of(es):
        out = []
        for k in range(len(es)):
            a, b = es[k - 1], es[k]
            out.append((a.c if a.axis == 0 else b.c, a.c if a.axis == 1 else b.c))
        return out

    def better(a: Edge, b: Edge) -> Edge:
        if a.plane is None:
            return b
        if b.plane is None:
            return a
        return a if a.plane.n >= b.plane.n else b

    # Remove degenerate edges: zero-length ones (two edges snapped onto the same wall line) and short
    # unsnapped notches (floor-boundary artefacts of furniture, not walls). Removing edge k makes its two
    # neighbours (same axis, since edges alternate) adjacent; merge them and recompute the corners.
    while len(merged) >= 4:
        cs = corners_of(merged)
        n = len(merged)
        lens = [np.hypot(cs[(k + 1) % n][0] - cs[k][0], cs[(k + 1) % n][1] - cs[k][1]) for k in range(n)]
        bad = [k for k in range(n) if lens[k] < 0.08 or (merged[k].plane is None and lens[k] < 0.3)]
        if not bad:
            break
        k = min(bad, key=lambda i: lens[i])
        a, b = merged[k - 1], merged[(k + 1) % n]
        keep = better(a, b)
        idx = sorted({(k - 1) % n, k, (k + 1) % n})
        new = [e for i, e in enumerate(merged) if i not in idx]
        insert_at = min(idx) if not (0 in idx and n - 1 in idx) else 0
        new.insert(min(insert_at, len(new)), keep)
        merged = new
    if len(merged) < 4:
        return None
    corners = corners_of(merged)
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


def _edges_of(poly: Polygon, old: list[Edge]) -> tuple[list, list[Edge]] | None:
    """Corners (CCW) and edges of a rectilinear polygon, reusing the old edge on each side where one
    lies on the same line; any other side lies on a neighbour's boundary (shared, no plane of its own)."""
    poly = poly.simplify(1e-6)
    if not poly.exterior.is_ccw:
        poly = Polygon(list(poly.exterior.coords)[::-1])
    pts = [tuple(map(float, c)) for c in poly.exterior.coords[:-1]]
    n = len(pts)
    if n < 4:
        return None
    edges: list[Edge] = []
    for k in range(n):
        p, q = np.array(pts[k]), np.array(pts[(k + 1) % n])
        d = q - p
        if min(abs(d[0]), abs(d[1])) > 1e-6:
            return None                        # not rectilinear
        axis = 0 if abs(d[0]) < abs(d[1]) else 1
        c = float(p[axis])
        lo, hi = sorted((p[1 - axis], q[1 - axis]))
        # CCW: the interior is on the left of p -> q, and the wall faces into the room
        face = ("+" if (axis == 0 and d[1] < 0) or (axis == 1 and d[0] > 0) else "-") + "xy"[axis]
        match = [e for e in old if e.axis == axis and abs(e.c - c) < 1e-6 and e.face == face]
        edges.append(match[0] if match else Edge(axis, c, face, None, shared=True))
    return pts, edges


def reconcile_rooms(polys: list[tuple[list, list[Edge]]], max_loss: float = 0.3):
    """Rooms are a partition, so their polygons must not overlap; each was snapped to walls on its own.
    For every overlapping pair the room whose intruding edges are backed by the better-supported wall
    plane keeps the contested strip; the other is clipped to it and takes that line as a shared
    boundary. Two observed wall faces that cross (a wall of negative thickness) are reported: that is
    pose drift between the two sides, not geometry. Returns (polys, warnings)."""
    from shapely.geometry import Point
    polys = list(polys)
    warns = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i] is None or polys[j] is None:
                continue
            Pi, Pj = Polygon(polys[i][0]), Polygon(polys[j][0])
            ov = Pi.intersection(Pj)
            if ov.area < 1e-3:
                continue

            def claim(P, edges, other):
                """Support of P's edges that lie inside the other room (the edges that intrude)."""
                cs = list(P.exterior.coords)
                best = 0
                for k, e in enumerate(edges):
                    mid = Point((cs[k][0] + cs[k + 1][0]) / 2, (cs[k][1] + cs[k + 1][1]) / 2)
                    if other.buffer(1e-6).contains(mid) and not other.exterior.buffer(1e-6).contains(mid):
                        best = max(best, e.plane.n if e.plane is not None and not e.shared else 0)
                return best

            ci, cj = claim(Pi, polys[i][1], Pj), claim(Pj, polys[j][1], Pi)
            win, lose = (i, j) if ci >= cj else (j, i)
            Pw, Pl = (Pi, Pj) if win == i else (Pj, Pi)
            minx, miny, maxx, maxy = ov.bounds
            if ci > 0 and cj > 0:
                warns.append(f"R{i + 1}/R{j + 1}: their wall faces cross by {100 * min(maxx - minx, maxy - miny):.0f} cm "
                             "(a wall cannot be thinner than zero: pose drift between the two sides, or one face is not "
                             "the wall); the better-supported face is kept")
            rest = Pl.difference(Pw)
            if rest.geom_type != "Polygon":
                rest = max(getattr(rest, "geoms", [rest]), key=lambda g_: g_.area)
            if rest.is_empty or rest.area < (1 - max_loss) * Pl.area:
                warns.append(f"R{i + 1}/R{j + 1}: outlines overlap by {ov.area:.2f} m2 and could not be reconciled")
                continue
            r = _edges_of(rest, polys[lose][1])
            if r is None:
                warns.append(f"R{i + 1}/R{j + 1}: outlines overlap by {ov.area:.2f} m2 and could not be reconciled")
                continue
            polys[lose] = r
    return polys, warns


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
    observed: bool = True    # snapped to an observed wall plane (openings are only searched on these)


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
    head_observed: bool = True   # wall seen above the opening; else its top is only a lower bound


def opening_evidence(cap, frame_corr, faces: list[Face], room_polys: list[Polygon], floor: HPlane,
                     frame_step: int = 2, band: float = 0.04, beyond: float = 0.25):
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


def _path_crosses(cams_xy: np.ndarray, f: Face, u0: float, u1: float) -> bool:
    """Did the camera path cross face f's line between u0 and u1 (walk through the opening)?"""
    if cams_xy is None or len(cams_xy) < 2:
        return False
    k = f.axis
    d = cams_xy[:, k] - f.offset
    idx = np.flatnonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0)
    for i in idx:
        t = d[i] / (d[i] - d[i + 1])
        u = cams_xy[i, 1 - k] + t * (cams_xy[i + 1, 1 - k] - cams_xy[i, 1 - k])
        if u0 <= u <= u1:
            return True
    return False


def detect_openings(grids, faces: list[Face], cams_xy: np.ndarray | None = None) -> list[OpeningEst]:
    """Openings need POSITIVE evidence: depth seen through the wall plane (>= 25 cm beyond it, set in
    opening_evidence), outnumbering solid returns over most of the gap. A gap that is only occlusion or
    no data has no through-points and is not an opening. Only faces snapped to an observed
    wall are searched: on an inferred edge there is no wall to have an opening in.

    Door or window, also from positive evidence (config/layout.yaml openings): the bottom of a doorway
    is the part least seen through (from chest height only steep rays pass it and land beyond), so a
    missing through-band at the floor is not a sill. A window is a gap with wall seen below it; a door
    is a gap the camera walked through, or one with no wall seen below it whose head is at door height."""
    ocfg = layout_config().get("openings", {})
    sill_frac, head_min = ocfg.get("sill_solid_frac", 0.5), ocfg.get("door_head_min_m", 1.8)
    bottom_max, width_max = ocfg.get("door_bottom_max_m", 0.6), ocfg.get("door_width_max_m", 1.6)
    out = []
    for G, f in zip(grids, faces):
        if not f.observed:
            continue
        S, T = G["solid"], G["thru"]
        # glass returns some solid points too, so through-points need only outnumber them
        openm = ((T >= 3) & (T > S)).astype(np.uint8)
        openm = cv2.morphologyEx(openm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        openm = cv2.morphologyEx(openm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(openm, connectivity=4)
        for k in range(1, n):
            x, y, w, h, area = stats[k]
            if w * GRID < 0.35 or h * GRID < 0.3 or area < 0.6 * w * h * 0.5:
                continue
            # positive evidence over most of the gap, not a few stray rays
            if (T[y:y + h, x:x + w] >= 3).mean() < 0.4:
                continue
            u0c, u1c = f.a + x * GRID, f.a + (x + w) * GRID
            v0c, v1c = y * GRID, (y + h) * GRID
            touches_floor = v0c < 0.15
            below = S[:y, x:x + w]
            sill_seen = below.size > 0 and float((below.sum(0) >= 3).mean()) >= sill_frac
            doorlike = (touches_floor or _path_crosses(cams_xy, f, u0c, u1c)
                        or (not sill_seen and v1c >= head_min and v0c <= bottom_max))
            if touches_floor and v1c > f.height - 0.08:
                typ = "opening"
            elif doorlike:
                typ = "door" if (u1c - u0c) <= width_max else "opening"
            else:
                typ = "window"
            mid0, mid1 = v0c + 0.1 * (v1c - v0c), v1c - 0.1 * (v1c - v0c)
            u0, s0 = _refine_edge(G["pts"], u0c, -1, mid0, mid1, along=0)
            u1, s1 = _refine_edge(G["pts"], u1c, +1, mid0, mid1, along=0)
            # vertical edges: head (solid above) and sill (solid below)
            cu0, cu1 = u0 + 0.1 * (u1 - u0), u1 - 0.1 * (u1 - u0)
            pts_vu = G["pts"][:, ::-1] if len(G["pts"]) else G["pts"]
            if typ == "opening" and v1c > f.height - 0.08:     # full height: no head to measure
                v1, s3 = f.height, 0.0
            else:
                v1, s3 = _refine_edge(pts_vu, v1c, +1, cu0, cu1, along=0)
            if typ == "window":
                v0, s2 = _refine_edge(pts_vu, v0c, -1, cu0, cu1, along=0)
            else:
                v0, s2 = 0.0, 0.0
            frac = (T[y:y + h, x:x + w].sum() + 1) / (T[y:y + h, x:x + w].sum() + S[y:y + h, x:x + w].sum() + 1)
            # the top is measured only if wall was seen above it (a floor-only scan sees walls to ~1.3 m,
            # so the see-through region of a door simply stops where the scan stopped)
            above = S[y + h:, x:x + w]
            head_seen = (typ == "opening" and v1c > f.height - 0.08) or \
                (above.size > 0 and float((above.sum(0) >= 3).mean()) >= sill_frac)
            out.append(OpeningEst(f, typ, u0, u1, v0, v1, float(frac),
                                  float(np.hypot(s0, s1)), float(np.hypot(s2, s3)), head_seen))
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
