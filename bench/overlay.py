"""Debug view: top-down slice of the fused cloud (wall height band) under the result polygons.

If the polygon edges do not sit on dense point lines, layout fitting is wrong regardless of what the
numbers say. Usage: python bench/overlay.py out/<capture>
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

out = Path(sys.argv[1])
r = json.loads((out / "result.json").read_text())
c = np.load(out / "cloud.npz")
P, N = c["P"], c["N"]
up = N[:, 2] > 0.9
floor_z = np.percentile(P[up, 2], 5)   # cloud is stored floor-unshifted; heights below are relative to floor
band = (P[:, 2] - floor_z > 0.8) & (P[:, 2] - floor_z < 1.8)
wallish = np.abs(N[:, 2]) < 0.3
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(P[band & ~wallish, 0], P[band & ~wallish, 1], s=0.3, c="0.75", label="other points 0.8-1.8 m")
ax.scatter(P[band & wallish, 0], P[band & wallish, 1], s=0.5, c="k", label="wall-facing points")
for rm in r["rooms"]:
    poly = np.array(rm["polygon"] + rm["polygon"][:1])
    ax.plot(poly[:, 0], poly[:, 1], "r-", lw=1.5)
ax.set_aspect("equal")
ax.legend(loc="upper right")
ax.set_title(f"{r['meta']['capture_id']}: cloud slice vs fitted polygon (red)")
fig.savefig(out / "overlay.png", dpi=110, bbox_inches="tight")
print(out / "overlay.png")
