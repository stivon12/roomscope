"""Room partition by structural-clearance persistence (SCP).

A room is a region of free floor that is much wider inside than the passage joining it to its
neighbours. With D = distance from each free cell to the nearest obstacle, a room centre is a maximum of
D (radius r_b) and a doorway is a saddle of D (half the door width, r_s). The 0-dimensional persistence
of the superlevel sets of L = log D gives, for every maximum, p = log(r_b / r_s): how much wider the
region is than its narrowest exit. It is scale-free:

    bedroom (r_b 1.5 m, 0.8 m door)  p = log 3.75
    bathroom                          p ~ log 2.3
    L-shaped room, alcove             p ~ log 1.0-1.4
    gap beside a wardrobe             no maximum of its own

Regions whose maximum has p > tau and r_b >= r_min seed a watershed of -L, so boundaries fall on the
bottlenecks. The decision is global (it compares the whole region with its narrowest exit), unlike a
gap-by-gap doorway rule. tau is chosen from data in the band set in config/layout.yaml, which was
calibrated on Bormann et al.'s labelled maps (bench/room_seg_eval.py) before use on our captures.
Each boundary carries a margin p - tau (docs/ROOM_SEGMENTATION.md).

References: Bormann et al., "Room segmentation: survey, implementation, and analysis", ICRA 2016
(distance-transform family); Edelsbrunner & Harer, Computational Topology (0-dim persistence); Chazal
et al., ToMATo, J. ACM 2013 (persistence-based clustering with gap selection of the threshold).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Peak:
    ij: tuple[int, int]      # cell of the maximum
    r_birth: float           # clearance at the maximum (m)
    r_death: float           # clearance at the saddle where it merges into an older region (m); 0 for the oldest
    @property
    def persistence(self) -> float:
        return float("inf") if self.r_death <= 0 else float(np.log(self.r_birth / self.r_death))


@dataclass
class Partition:
    labels: np.ndarray       # int32, 0 outside the free mask, 1..k rooms
    peaks: list[Peak]        # all maxima with their persistence (for diagnostics and threshold choice)
    seeds: list[Peak]        # maxima that became rooms
    tau: float               # persistence threshold used (log ratio)
    margins: list[float]     # persistence - tau of each seed except the oldest (how clear each split is)


def clearance(free: np.ndarray, cell_m: float) -> np.ndarray:
    """Distance (m) from each free cell to the nearest non-free cell."""
    return ndimage.distance_transform_edt(free) * cell_m


def persistence_peaks(L: np.ndarray, free: np.ndarray, floor_log: float) -> list[Peak]:
    """0-dim persistence of superlevel sets of L over the free cells (4-connectivity, union-find).
    A region is born at a local maximum and dies when it merges into a region with a higher maximum
    (elder rule); the merge level is the saddle."""
    H, W = L.shape
    idx = np.flatnonzero(free & (L > floor_log))
    order = idx[np.argsort(-L.ravel()[idx], kind="stable")]
    parent = np.full(H * W, -1, np.int64)
    peak_of = {}                      # root -> flat index of its maximum
    flatL = L.ravel()
    peaks: list[Peak] = []

    def find(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r

    for c in order:
        parent[c] = c
        peak_of[c] = c
        i, j = divmod(int(c), W)
        roots = set()
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= ni < H and 0 <= nj < W:
                n = ni * W + nj
                if parent[n] >= 0:
                    roots.add(find(n))
        if not roots:
            continue
        roots = sorted(roots, key=lambda r: -flatL[peak_of[r]])
        keep = roots[0]
        for r in roots[1:]:            # younger regions die here, at this cell's level
            pk = peak_of[r]
            peaks.append(Peak(divmod(int(pk), W), float(np.exp(flatL[pk])), float(np.exp(flatL[c]))))
            parent[r] = keep
        parent[c] = keep
    for r in {find(int(c)) for c in order}:   # survivors (one per connected component) never die
        pk = peak_of[r]
        peaks.append(Peak(divmod(int(pk), W), float(np.exp(flatL[pk])), 0.0))
    return peaks


def choose_tau(peaks: list[Peak], band: tuple[float, float], r_min: float) -> float:
    """Threshold at the widest gap between sorted persistence values that falls inside `band`
    (log ratios); the band centre when no value falls inside it (nothing ambiguous to separate)."""
    lo, hi = band
    p = np.sort([pk.persistence for pk in peaks if np.isfinite(pk.persistence) and pk.r_birth >= r_min])
    cand = np.concatenate([[lo], p[(p > lo) & (p < hi)], [hi]])
    if len(cand) <= 2:
        return 0.5 * (lo + hi)
    gaps = np.diff(cand)
    k = int(np.argmax(gaps))
    return float(0.5 * (cand[k] + cand[k + 1]))


def partition(free: np.ndarray, cell_m: float, band: tuple[float, float], r_min: float,
              r_floor: float = 0.05, min_area_m2: float = 0.0) -> Partition:
    """Partition the free mask into rooms. band: (lo, hi) of tau as log ratios; r_min: smallest room
    'radius' (m) that can seed a room; r_floor: clearance below which cells are not used for peaks."""
    from skimage.segmentation import watershed
    D = clearance(free, cell_m)
    L = np.log(np.maximum(D, r_floor))
    peaks = persistence_peaks(L, free, np.log(r_floor))
    tau = choose_tau(peaks, band, r_min)
    seeds = [pk for pk in peaks if pk.r_birth >= r_min and pk.persistence > tau]
    if not seeds and peaks:
        seeds = [max(peaks, key=lambda pk: pk.r_birth)]
    markers = np.zeros(free.shape, np.int32)
    for k, pk in enumerate(seeds, 1):
        markers[pk.ij] = k
    labels = watershed(-L, markers, mask=free) if seeds else np.zeros(free.shape, np.int32)
    if min_area_m2 > 0:              # tiny leftovers (cut off by noise) are not rooms
        area = np.bincount(labels.ravel()) * cell_m * cell_m
        labels[np.isin(labels, np.flatnonzero(area < min_area_m2))] = 0
    margins = [pk.persistence - tau for pk in seeds if np.isfinite(pk.persistence)]
    return Partition(labels.astype(np.int32), peaks, seeds, tau, margins)
