"""Plane-anchored drift correction (the brief's 'drift accountability' row).

Why this design: odometry (ARKit VIO) drifts slowly in yaw and translation over a multi-room walk, so the
same physical wall seen at minute 0 and minute 2 lands in two different places and the fused plan smears.
Interiors give us strong, cheap landmarks that do not drift: the floor, and large wall planes aligned to
a few dominant (Manhattan) directions. So:

1. Split the trajectory into short time fragments (default 4 s). Within a fragment drift is negligible.
2. Yaw: each fragment's Manhattan direction is observable directly from its wall normals; rotate the
   fragment so its walls align with the global axes (removes accumulated heading drift).
3. Translation: extract each fragment's wall planes (face direction + offset) and floor height. Walk
   fragments in time order, anchoring each to a growing global map of planes; a fragment that revisits a
   wall seen long ago (returning through a doorway, ending where we started) is pulled onto it - this is
   the loop closure.
4. Joint robust least squares over all fragment translations and global plane offsets (fragment 0 fixed),
   with the associations from step 3; soft-L1 loss so one bad association cannot drag the solution.
5. Per-frame corrections are linearly interpolated between fragment centres (no seams).

Defaults (2026-10-04, docs/DIAGNOSTICS.md): no per-fragment yaw, 8 cm association, 3 cm odometry prior.
Scored per frame against the Faro laser, the original settings (per-fragment yaw, 20 cm association, no
prior) made ARKit poses worse: median camera error 2.4 -> 8.2 cm on 42444946. The defaults do no harm
(42444949: 1.9 -> 1.9 cm, 42444950: 2.2 -> 1.9 cm). KNOWN LIMIT: they remove little of a large injected
drift (bench/drift_inject.py: max 16.6 -> 15.5 cm). Plane merging removes about half of it but adds
2-4 cm to accurate poses, because a 4 s fragment's plane observations are no more precise than ARKit.

Ablation: `--no-drift` returns identity corrections; the report compares plane-consistency residuals and
the stitched footprint with and without.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .layout import FACES, classify, fit_hplane, manhattan_yaw, pick_level, rz

MATCH_DIST = 0.20      # m: max offset disagreement to associate a fragment plane with a map plane
EXTENT_SLACK = 1.0     # m: along-wall extents must overlap within this slack


@dataclass
class PlaneObs:
    frag: int
    face: str
    axis: int
    offset: float
    lo: float
    hi: float
    w: float           # weight ~ sqrt(n)/std


def _frag_planes(P, N, frag, min_pts=150, floor_z=None):
    """Wall-plane observations in one fragment. With floor_z given, only STRUCTURAL planes are kept
    (research note 2026-10-04, after Choi/Zhou/Koltun CVPR 2015 and Manhattan-SLAM practice: gate
    landmarks before solving, so a wardrobe face can never be associated with a wall):
    - tall: the plane's points span >= 1.0 m vertically and reach >= 1.6 m above the floor;
    - wide: >= 0.8 m along the wall;
    - outermost: no other plane of the same face lies > 5 cm further out with >= 30% overlap along it."""
    cls = classify(N)
    obs = []
    for face, (axis, _sign) in FACES.items():
        m = cls == face
        if m.sum() < min_pts:
            continue
        x = P[m, axis]
        along = P[m, 1 - axis]
        edges = np.arange(x.min() - 0.02, x.max() + 0.03, 0.02)
        h, _ = np.histogram(x, edges)
        used = np.zeros(len(h), bool)
        for pk in np.argsort(h)[::-1][:6]:
            if h[pk] < min_pts or used[max(0, pk - 4):pk + 5].any():
                continue
            used[pk] = True
            off, win = edges[pk] + 0.01, 0.05
            for _ in range(3):
                sel = np.abs(x - off) < win
                off = float(np.mean(x[sel]))
                win = max(2.5 * np.std(x[sel] - off), 0.015)
            sel = np.abs(x - off) < win
            if sel.sum() < min_pts:
                continue
            a = along[sel]
            std = max(float(np.std(x[sel] - off)), 0.005)
            lo, hi = float(np.percentile(a, 2)), float(np.percentile(a, 98))
            if floor_z is not None:
                z = P[m, 2][sel]
                z_lo, z_hi = np.percentile(z, [2, 98])
                if z_hi - z_lo < 1.0 or z_hi - floor_z < 1.6 or hi - lo < 0.8:
                    continue
            obs.append(PlaneObs(frag, face, axis, off, lo, hi, np.sqrt(sel.sum()) / std))
    if floor_z is not None:
        keep = []
        for o in obs:
            out = FACES[o.face][1]          # normals point into the room: outward is -sign along the axis
            hidden = any(q is not o and q.face == o.face and (o.offset - q.offset) * out > 0.05 and
                         (min(o.hi, q.hi) - max(o.lo, q.lo)) >= 0.3 * (o.hi - o.lo) for q in obs)
            if not hidden:
                keep.append(o)
        obs = keep
    return obs


@dataclass
class DriftResult:
    corrections: np.ndarray        # (F,4,4) world corrections to left-multiply onto poses
    residual_before: float         # weighted RMS plane disagreement (m), raw poses
    residual_after: float
    n_fragments: int
    n_map_planes: int
    method: str = "plane-anchored fragment alignment + robust joint LSQ (wall/floor planes, 3 cm odometry prior)"


def correct_drift(cap, frag_seconds: float = 4.0, frame_step: int = 2, per_fragment_yaw: bool | str = False,
                  match_dist: float = 0.08, prior_sigma: float | None = 0.03,
                  merge_dist: float = 0.0, reassoc_iters: int = 0, structural: bool = False,
                  loss: str = "soft_l1", prior_mode: str = "absolute") -> DriftResult:
    """per_fragment_yaw: True = each fragment's own Manhattan yaw (noisy, ~1 deg); "linear" = one robust
    linear yaw trend over the capture fitted to those per-fragment estimates; False = no yaw correction.
    merge_dist/reassoc_iters: after the joint solve, merge map planes of the same face whose solved
    offsets agree within merge_dist (with overlapping extents) and re-solve, so revisiting a wall
    actually constrains the drift instead of spawning a duplicate plane that absorbs it.
    structural: keep only tall, wide, outermost wall planes as landmarks (see _frag_planes).
    loss: scipy robust loss for the joint solve ("soft_l1", or the stronger "cauchy").
    prior_mode: "absolute" penalises each fragment's correction (|t_k| ~ prior_sigma): safe, but it caps
    the correction near prior_sigma, so a 15 cm drift can never be removed. "relative" is the pose-graph
    odometry edge (Choi et al. 2015): trust ARKit BETWEEN consecutive fragments (|t_k - t_k-1| ~
    prior_sigma), so slow drift may accumulate into a large correction."""
    F = len(cap.poses)
    ts = cap.timestamps - cap.timestamps[0]
    frag_id = np.floor(ts / frag_seconds).astype(int)
    K = frag_id.max() + 1

    # global reference axes from the whole (drifted) cloud
    allN = np.concatenate([cap.world(i)[1] for i in range(0, F, 5)])
    R0 = rz(-manhattan_yaw(allN))
    R0i = np.linalg.inv(R0)

    # pass 1: per-fragment yaw + pivot
    theta = np.zeros(K)
    pivot = np.zeros((K, 3))
    frag_pts = {}
    for k in range(K):
        idx = np.where(frag_id == k)[0][::frame_step]
        if len(idx) == 0:
            continue
        Ps, Ns, cs = [], [], []
        for i in idx:
            P, N, c = cap.world(i, R0)
            Ps.append(P); Ns.append(N); cs.append(c)
        P, N = np.concatenate(Ps), np.concatenate(Ns)
        pivot[k] = np.mean(cs, axis=0)
        if per_fragment_yaw and (np.abs(N[:, 2]) < 0.3).sum() > 500:
            theta[k] = -manhattan_yaw(N)
        frag_pts[k] = (P, N)

    if per_fragment_yaw == "linear":
        have = np.array([k in frag_pts and (np.abs(frag_pts[k][1][:, 2]) < 0.3).sum() > 500 for k in range(K)])
        kk = np.arange(K)[have]
        th = theta[have]
        th = (th + np.pi / 4) % (np.pi / 2) - np.pi / 4      # wrap to +-45 deg
        if len(kk) >= 4:
            w = np.ones(len(kk))
            for _ in range(5):                                # Huber IRLS on theta = a + b k
                A = np.c_[np.ones(len(kk)), kk] * np.sqrt(w)[:, None]
                ab = np.linalg.lstsq(A, th * np.sqrt(w), rcond=None)[0]
                r = np.abs(th - (ab[0] + ab[1] * kk))
                w = np.where(r < np.deg2rad(0.5), 1.0, np.deg2rad(0.5) / np.maximum(r, 1e-9))
            # apply the trend only if it is clearly real: total change > max(3 SE, 0.5 deg). On real
            # ARKit poses an ungated trend added ~1 deg of yaw error (Manhattan yaw from a fragment's
            # visible walls is biased by which walls are in view, not just by drift)
            res = th - (ab[0] + ab[1] * kk)
            se_b = np.sqrt(np.sum(w * res ** 2) / max(len(kk) - 2, 1) / np.sum(w * (kk - kk.mean()) ** 2))
            span = K - 1
            if abs(ab[1]) * span > max(3 * se_b * span, np.deg2rad(0.5)):
                theta = ab[1] * np.arange(K)                  # gauge: fragment 0 keeps its yaw
            else:
                theta = np.zeros(K)
        else:
            theta = np.zeros(K)

    def frag_M(k, t):
        """Correction in the R0 frame: rotate by theta about pivot, then translate by t."""
        M = np.eye(4)
        Rk = rz(theta[k])
        M[:3, :3] = Rk[:3, :3]
        M[:3, 3] = pivot[k] - Rk[:3, :3] @ pivot[k] + t
        return M

    # pass 2: per-fragment planes after yaw correction
    obs: list[PlaneObs] = []
    floor_z = np.full(K, np.nan)
    floor_w = np.zeros(K)
    frag_floor = {}
    for k, (P, N) in frag_pts.items():
        M = frag_M(k, np.zeros(3))
        P2 = P @ M[:3, :3].T + M[:3, 3]
        N2 = N @ M[:3, :3].T
        fl = N2[:, 2] > 0.9
        if fl.sum() > 300:
            # lowest substantial level, not the median: in a bedroom fragment the bed top often wins
            hp = fit_hplane(P2[fl], z0=pick_level(P2[fl], "low", min_extent_m2=0.5), iters=2)
            floor_z[k], floor_w[k] = hp.c, np.sqrt(hp.n) / max(hp.std, 0.005)
        frag_floor[k] = floor_z[k]
    if structural:
        # the gate uses the capture's median floor (ARKit z drifts by cm only)
        fz = float(np.nanmedian(floor_z)) if (~np.isnan(floor_z)).any() else float(np.min(cap.poses[:, 2, 3]) - 1.4)
        frag_floor = {k: fz for k in frag_floor}    # one floor for the gate: a fragment's own may be a bed top
    for k, (P, N) in frag_pts.items():
        M = frag_M(k, np.zeros(3))
        obs += _frag_planes(P @ M[:3, :3].T + M[:3, 3], N @ M[:3, :3].T, k,
                            floor_z=frag_floor[k] if structural else None)

    # pass 3: sequential anchoring to a growing map (association + initial translations)
    t = np.zeros((K, 3))
    gmap: list[dict] = []          # {face, axis, g, lo, hi, wsum}
    assoc: list[tuple[int, int]] = []   # (obs index, map index)
    by_frag = {}
    for oi, o in enumerate(obs):
        by_frag.setdefault(o.frag, []).append(oi)
    prev = np.zeros(3)
    floor_ref = float(np.nanmedian(floor_z)) if (~np.isnan(floor_z)).any() else 0.0
    # ARKit's vertical drift over a few minutes is centimetres; a fragment whose "floor" is >10 cm off the
    # median is a misdetection (rug, bed, step), so it gets no height anchor rather than a wrong one
    bad = np.abs(floor_z - floor_ref) > 0.10
    floor_z[bad] = np.nan
    floor_w[bad] = 0.0
    for k in range(K):
        t[k] = prev
        ois = by_frag.get(k, [])
        matches = []
        for oi in ois:
            o = obs[oi]
            best, bd = None, match_dist
            for mi, g in enumerate(gmap):
                if g["face"] != o.face:
                    continue
                if o.lo > g["hi"] + EXTENT_SLACK or o.hi < g["lo"] - EXTENT_SLACK:
                    continue
                d = abs(o.offset + t[k, o.axis] - g["g"])
                if d < bd:
                    best, bd = mi, d
            matches.append((oi, best))
        for axis in (0, 1):
            diffs = [(gmap[mi]["g"] - obs[oi].offset, obs[oi].w) for oi, mi in matches
                     if mi is not None and obs[oi].axis == axis]
            if diffs:
                d, w = np.array(diffs).T
                order = np.argsort(d)
                cw = np.cumsum(w[order])
                t[k, axis] = d[order][np.searchsorted(cw, cw[-1] / 2)]   # weighted median
        if not np.isnan(floor_z[k]):
            t[k, 2] = floor_ref - floor_z[k]
        for oi, mi in matches:
            o = obs[oi]
            # re-check association with the updated translation; unmatched planes extend the map
            if mi is not None and abs(o.offset + t[k, o.axis] - gmap[mi]["g"]) < match_dist:
                g = gmap[mi]
                g["g"] = (g["g"] * g["wsum"] + (o.offset + t[k, o.axis]) * o.w) / (g["wsum"] + o.w)
                g["wsum"] += o.w
                g["lo"], g["hi"] = min(g["lo"], o.lo), max(g["hi"], o.hi)
                assoc.append((oi, mi))
            else:
                gmap.append({"face": o.face, "axis": o.axis, "g": o.offset + t[k, o.axis],
                             "lo": o.lo, "hi": o.hi, "wsum": o.w})
                assoc.append((oi, len(gmap) - 1))
        prev = t[k].copy()

    # pass 4: joint robust least squares (fragment 0 fixed: gauge)
    M_ = len(gmap)

    def unpack(x):
        tt = np.zeros((K, 3))
        tt[1:] = x[:3 * (K - 1)].reshape(K - 1, 3)
        return tt, x[3 * (K - 1):]

    wmax = max(o.w for o in obs) if obs else 1.0
    frag_w = np.zeros(K)
    for oi, _ in assoc:
        frag_w[obs[oi].frag] += obs[oi].w / wmax

    def resid(x):
        tt, g = unpack(x)
        r = [np.sqrt(obs[oi].w / wmax) * (obs[oi].offset + tt[obs[oi].frag, obs[oi].axis] - g[mi]) for oi, mi in assoc]
        fw = floor_w / (floor_w.max() if floor_w.max() > 0 else 1)
        r += [np.sqrt(fw[k]) * (floor_z[k] + tt[k, 2] - floor_ref) for k in range(K) if not np.isnan(floor_z[k])]
        # weak smoothness prior: drift is slow, so neighbouring fragments should not jump
        r += list(0.05 * (tt[1:] - tt[:-1]).ravel())
        if prior_sigma is not None:
            # trust the odometry: a fragment moves only when its planes disagree by clearly more than
            # ARKit's typical drift (prior_sigma); the prior weighs as much as the fragment's own planes
            if prior_mode == "relative":
                fw_ = np.sqrt(np.minimum(frag_w[1:], frag_w[:-1]) + 1e-9)[:, None]
                r += list((fw_ * (tt[1:] - tt[:-1]) * (0.02 / prior_sigma)).ravel())
            else:
                r += list((np.sqrt(frag_w[1:])[:, None] * tt[1:] * (0.02 / prior_sigma)).ravel())
        return np.asarray(r)

    x0 = np.concatenate([t[1:].ravel(), np.array([g["g"] for g in gmap])])
    sol = least_squares(resid, x0, loss=loss, f_scale=0.02) if len(x0) else None
    tt, g = unpack(sol.x) if sol is not None else (t, np.array([]))

    for _ in range(reassoc_iters if merge_dist > 0 else 0):
        # merge map planes: same face, solved offsets within merge_dist, extents overlapping
        order = sorted(range(len(gmap)), key=lambda m: (gmap[m]["face"], g[m]))
        root = list(range(len(gmap)))
        for a_, b_ in zip(order[:-1], order[1:]):
            A_, B_ = gmap[a_], gmap[b_]
            if A_["face"] == B_["face"] and abs(g[a_] - g[b_]) < merge_dist and \
                    A_["lo"] <= B_["hi"] + EXTENT_SLACK and B_["lo"] <= A_["hi"] + EXTENT_SLACK:
                root[b_] = root[a_]
        for m in range(len(root)):                     # path compression
            while root[root[m]] != root[m]:
                root[m] = root[root[m]]
        uniq = sorted(set(root))
        remap = {r: i for i, r in enumerate(uniq)}
        new_g = np.zeros(len(uniq)); new_w = np.zeros(len(uniq))
        new_map = []
        for m in range(len(gmap)):
            i = remap[root[m]]
            new_g[i] += g[m] * gmap[m]["wsum"]; new_w[i] += gmap[m]["wsum"]
        for r in uniq:
            members = [m for m in range(len(gmap)) if root[m] == r]
            new_map.append({"face": gmap[r]["face"], "axis": gmap[r]["axis"],
                            "lo": min(gmap[m]["lo"] for m in members), "hi": max(gmap[m]["hi"] for m in members),
                            "wsum": sum(gmap[m]["wsum"] for m in members)})
        merged = len(uniq) < len(gmap)
        gmap = new_map
        g = new_g / np.maximum(new_w, 1e-12)
        assoc = [(oi, remap[root[mi]]) for oi, mi in assoc]
        M_ = len(gmap)
        if not merged:
            break
        sol = least_squares(resid, np.concatenate([tt[1:].ravel(), g]), loss=loss, f_scale=0.02)
        tt, g = unpack(sol.x)

    def rms_free(tvec):
        """Weighted RMS plane disagreement with plane offsets re-fitted for these translations."""
        if not assoc:
            return float("nan")
        v = np.array([obs[oi].offset + tvec[obs[oi].frag, obs[oi].axis] for oi, _ in assoc])
        w = np.array([obs[oi].w for oi, _ in assoc])
        mi_ = np.array([mi for _, mi in assoc])
        e = np.zeros(len(v))
        for m in np.unique(mi_):
            sel = mi_ == m
            e[sel] = v[sel] - np.average(v[sel], weights=w[sel])
        return float(np.sqrt(np.average(e ** 2, weights=w)))

    disagree_raw, disagree_fit = rms_free(np.zeros((K, 3))), rms_free(tt)

    def rms(tvec, gvec):
        if not assoc:
            return float("nan")
        e = np.array([obs[oi].offset + tvec[obs[oi].frag, obs[oi].axis] - gvec[mi] for oi, mi in assoc])
        w = np.array([obs[oi].w for oi, _ in assoc])
        return float(np.sqrt(np.average(e ** 2, weights=w)))

    # "before": raw poses (no yaw, no translation) - recompute planes in the R0 frame with the same associations
    obs_raw = {}
    for k, (P, N) in frag_pts.items():
        for o in _frag_planes(P, N, k, floor_z=frag_floor[k] if structural else None):
            obs_raw.setdefault((k, o.face), []).append(o)
    e_b, w_b = [], []
    members: dict[int, list[float]] = {}
    for oi, mi in assoc:
        o = obs[oi]
        cands = obs_raw.get((o.frag, o.face), [])
        if not cands:
            continue
        r = min(cands, key=lambda c: abs(c.offset - o.offset))
        members.setdefault(mi, []).append((r.offset, r.w))
    for mi, lst in members.items():
        if len(lst) < 2:
            continue
        v, w = np.array(lst).T
        mu = np.average(v, weights=w)
        e_b += list(v - mu); w_b += list(w)
    before = float(np.sqrt(np.average(np.square(e_b), weights=w_b))) if e_b else float("nan")
    after = rms(tt, g)

    # per-frame corrections: interpolate (theta, translation-about-origin) between fragment centres
    centres = np.array([np.mean(np.where(frag_id == k)[0]) if (frag_id == k).any() else np.nan for k in range(K)])
    Ms = [frag_M(k, tt[k]) for k in range(K)]
    th = np.array([theta[k] for k in range(K)])
    tr = np.array([M[:3, 3] for M in Ms])
    ok = ~np.isnan(centres)
    fi = np.arange(F)
    th_f = np.interp(fi, centres[ok], th[ok])
    tr_f = np.stack([np.interp(fi, centres[ok], tr[ok, d]) for d in range(3)], 1)
    corr = np.zeros((F, 4, 4))
    for i in range(F):
        M = rz(th_f[i])
        M[:3, 3] = tr_f[i]
        corr[i] = R0i @ M @ R0
    res = DriftResult(corr, before, after, int(K), M_)
    res.disagree_raw, res.disagree_fit = disagree_raw, disagree_fit
    return res


def identity(cap) -> np.ndarray:
    return np.repeat(np.eye(4)[None], len(cap.poses), 0)
