"""Structural-clearance persistence (core/rooms_scp.py) on synthetic free-space masks at 4 cm."""
from __future__ import annotations

import numpy as np

from roomscope.core import rooms_scp as S

CELL = 0.04
BAND = (np.log(1.5), np.log(2.5))


def box(mask, x0, y0, x1, y1):
    mask[int(y0 / CELL):int(y1 / CELL), int(x0 / CELL):int(x1 / CELL)] = True


def n_rooms(mask):
    return len(np.unique(S.partition(mask, CELL, BAND, 0.6).labels)) - 1


def test_two_rooms_joined_by_a_door_split():
    m = np.zeros((150, 260), bool)
    box(m, 0.2, 0.2, 4.2, 4.2)          # 4 x 4 m room
    box(m, 4.4, 0.2, 8.4, 3.2)          # 4 x 3 m room
    box(m, 4.2, 1.6, 4.4, 2.4)          # 0.8 m doorway through a 0.2 m wall
    assert n_rooms(m) == 2


def test_l_shaped_room_stays_one():
    m = np.zeros((150, 150), bool)
    box(m, 0.2, 0.2, 5.2, 2.7)
    box(m, 0.2, 2.7, 2.7, 5.2)
    assert n_rooms(m) == 1


def test_wardrobe_gap_does_not_split():
    m = np.zeros((150, 150), bool)
    box(m, 0.2, 0.2, 4.7, 4.2)
    m[int(0.2 / CELL):int(1.0 / CELL), int(2.0 / CELL):int(3.2 / CELL)] = False   # wardrobe against a wall
    assert n_rooms(m) == 1


def test_open_plan_arch_merges():
    m = np.zeros((150, 260), bool)
    box(m, 0.2, 0.2, 4.2, 4.2)
    box(m, 4.4, 0.2, 8.4, 4.2)
    box(m, 4.2, 0.8, 4.4, 3.6)          # 2.8 m arch: one open-plan space
    assert n_rooms(m) == 1


def test_overlapping_rooms_are_reconciled_to_the_supported_wall():
    from shapely.geometry import Polygon
    from roomscope.core import layout as L

    class P:            # stand-in wall plane: only support (n) is used
        def __init__(self, n):
            self.n = n
    def rect(x0, y0, x1, y1, planes):
        r = L._edges_of(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), [])
        corners, edges = r
        for e in edges:
            e.shared, e.plane = False, planes.get((e.axis, round(e.c, 3)))
        return corners, edges
    a = rect(0, 0, 3.0, 3, {(0, 3.0): None})                 # right side unsnapped (open boundary)
    b = rect(2.8, 0, 6, 3, {(0, 2.8): P(5000)})              # left side on an observed wall
    (ca, ea), (cb, eb) = L.reconcile_rooms([a, b])[0]
    A, B = Polygon(ca), Polygon(cb)
    assert A.intersection(B).area < 1e-9 and abs(B.area - 3.2 * 3) < 1e-9
    assert any(e.shared and e.axis == 0 and abs(e.c - 2.8) < 1e-9 for e in ea)
