"""Rendered plan: rooms, wall dimensions with intervals, openings, adjacency. Matplotlib, no styling deps."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OPENING_COLOR = {"door": "#8c5a2b", "window": "#2b7bb9", "opening": "#888888"}


def _fmt(m) -> str:
    half = (m["hi"] - m["lo"]) / 2
    return f"{m['value']:.2f}±{half * 100:.1f}cm" if m["unit"] == "m" else f"{m['value']:.2f}"


def render_plan(result: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 11))
    for room in result["rooms"]:
        P = np.array(room["polygon"] + room["polygon"][:1])
        ax.fill(P[:, 0], P[:, 1], color="#f3efe6", zorder=1)
        ax.plot(P[:, 0], P[:, 1], color="#222", lw=2.5, zorder=2)
        c = P[:-1].mean(0)
        ax.text(c[0], c[1], f"{room['name']}\n{room['floor_area']['value']:.2f} m²\nceil {_fmt(room['ceiling_height'])}",
                ha="center", va="center", fontsize=9, zorder=5)
        for w in room["walls"]:
            s, e = np.array(w["start"]), np.array(w["end"])
            mid = (s + e) / 2
            d = e - s
            n = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)   # CCW polygon: left normal points inward
            t = mid + 0.22 * n
            ang = np.degrees(np.arctan2(d[1], d[0]))
            if ang > 90 or ang < -90:
                ang += 180
            ax.text(t[0], t[1], _fmt(w["length"]), ha="center", va="center", rotation=ang, fontsize=7,
                    color="#444", zorder=5)
        walls = {w["id"]: w for w in room["walls"]}
        for o in room["openings"]:
            w = walls[o["wall_id"]]
            s, e = np.array(w["start"]), np.array(w["end"])
            u = (e - s) / max(np.linalg.norm(e - s), 1e-9)
            a = s + u * o["offset"]["value"]
            b = a + u * o["width"]["value"]
            ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=4, zorder=3)
            ax.plot([a[0], b[0]], [a[1], b[1]], color=OPENING_COLOR[o["type"]], lw=2, zorder=4,
                    ls="-" if o["type"] != "window" else "--")
            m = (a + b) / 2
            n = np.array([-u[1], u[0]])
            t = m - 0.15 * n
            ax.text(t[0], t[1], f"{o['type'][0].upper()} {o['width']['value']:.2f}", fontsize=6,
                    color=OPENING_COLOR[o["type"]], ha="center", va="center", zorder=5)
    meta = result["meta"]
    fp = result["footprint"]
    ax.set_title(f"{meta['capture_id']} · tier={meta['tier']} · footprint {fp['area']['value']:.2f} m² "
                 f"[{fp['area']['lo']:.2f}, {fp['area']['hi']:.2f}] · drift={'on' if meta['drift_correction']['enabled'] else 'off'}",
                 fontsize=10)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_xlabel("m")
    ax.set_ylabel("m")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
