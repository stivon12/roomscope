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
