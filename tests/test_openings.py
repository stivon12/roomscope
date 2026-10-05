"""Door/window evidence helpers (core/layout.detect_openings)."""
from __future__ import annotations

import numpy as np

from roomscope.core import layout as L


def test_path_crossing_is_detected_only_inside_the_opening():
    f = L.Face(room=0, wall_idx=0, axis=0, sign=1, offset=2.0, a=0.0, b=4.0, height=2.5)
    walk = np.array([[1.0, 1.5], [1.8, 1.5], [2.3, 1.6], [3.0, 1.6]])     # crosses x = 2 at y ~ 1.54
    assert L._path_crosses(walk, f, 1.2, 2.0)
    assert not L._path_crosses(walk, f, 2.5, 3.5)
    assert not L._path_crosses(walk[:2], f, 0.0, 4.0)                    # never reaches the wall
