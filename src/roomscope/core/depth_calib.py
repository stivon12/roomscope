"""LiDAR depth-scale correction: a configured per-device prior (used by the pipeline), and an
EXPERIMENTAL in-scan self-calibration that is not used (see the note at the end of this docstring).

In-scan LiDAR depth-scale self-calibration (CLAMS idea: Teichman, Miller & Thrun, RSS 2013).

The phone LiDAR reads short by a roughly constant fraction of range (about 1.2% on ARKitScenes iPads,
measured against the Faro laser in eval/depth_bias.py). Camera translation comes from visual-inertial
odometry, whose metric scale is set by the IMU and is independent of the LiDAR. So if depth is scaled
by a = measured / true, a fixed plane seen from distance D appears (1 - a) * D closer than it is, and it
seems to move as the camera moves toward or away from it. Wanting every plane to stay put gives a linear
least-squares problem in one unknown, lambda = 1 / a:

    coordinate of plane j seen from frame i = t_i + lambda * m_i  (= c_j for every i)

where t_i is the camera's coordinate along the plane normal and m_i is the signed camera-to-plane
offset that frame i's depth reports. The plane offsets c_j are nuisance parameters, removed by
centring each plane's observations. No ground truth is used.

What makes it identifiable is range diversity: the same plane must be seen from different distances.
Pose drift (a few cm, slow) is the main noise, and it is correlated in time, so the standard error comes
from a block bootstrap over time chunks.

Status (2026-10-04): NOT RELIABLE on ARKitScenes 42444946/49/50 and NOT used by the pipeline. It returned
+3.07%, +0.19% and +0.20% (laser truth: -1.31%, -1.10%, -1.23%). Short-window differencing and a joint
pose-latency fit did not fix it: estimates ranged from -2.0% to +3.9%. The signal is small: a 1.2% scale
error over the 0.2-0.4 m spread of camera-to-plane distance in these captures is only 2-5 mm, against
2-3 cm of slow ARKit pose drift (eval/pose_drift.py). Kept for a capture protocol with deliberate range
diversity.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .layout import manhattan_yaw, rz

CONFIG = __import__("pathlib").Path(__file__).resolve().parents[3] / "config" / "depth_scale.yaml"


@dataclass
class DepthScale:
    scale: float     # a = measured / true
    se: float
    source: str
    device: str = "unknown"
    excess_se: float = 0.0   # scale se beyond the calibration device's (added to intervals by core/calibrate)

    def to_json(self) -> dict:
        return {"enabled": self.scale != 1.0, "scale": round(self.scale, 5), "se": round(self.se, 5),
                "excess_se": round(self.excess_se, 5), "device": self.device, "source": self.source}


def load_config() -> dict:
    import yaml
    return yaml.safe_load(CONFIG.read_text())


def resolve_depth_scale(device: str | None = None, override: float | None = None,
                        enabled: bool = True) -> DepthScale:
    """Depth scale to apply: an explicit override (e.g. from a tape-measured distance), else the
    device's entry in config/depth_scale.yaml, else the unknown-device default."""
    if not enabled:
        return DepthScale(1.0, 0.0, "disabled (ablation: depth as recorded)", device or "unknown")
    if override is not None:
        return DepthScale(float(override), 0.0, "override (--depth-scale)", device or "unknown")
    cfg = load_config()
    devs = cfg.get("devices") or {}
    e = devs.get(device or "") or cfg["default"]
    ref = devs.get(cfg.get("calibration_device", ""), {})
    se = float(e["se"])
    excess = float(np.sqrt(max(0.0, se ** 2 - float(ref.get("se", 0.0)) ** 2)))
    return DepthScale(float(e["scale"]), se, str(e["source"]), device if device in devs else "unknown", excess)


@dataclass
class DepthCalib:
    scale: float            # a = measured / true depth; correct depth by dividing by it
    se: float               # bootstrap standard error of `scale`
    n_planes: int
    n_obs: int
    range_spread_m: float   # median per-plane std of camera-to-plane distance (identifiability)
    method: str = "selfcal:plane-consistency:v1"

    @property
    def rel(self) -> float:
        return self.scale - 1.0


def _peaks(h: np.ndarray, min_count: float, min_sep: int) -> list[int]:
    idx = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] >= min_count]
    idx.sort(key=lambda i: -h[i])
    keep: list[int] = []
    for i in idx:
        if all(abs(i - j) >= min_sep for j in keep):
            keep.append(i)
    return keep


def _observations(cap, frame_step: int, win: float, min_pts: int, cos_th: float = 0.95):
    """Per (frame, plane) observation: plane id, camera coordinate t, median reported offset m, time."""
    F = len(cap.poses)
    frames = list(range(0, F, frame_step))
    # Manhattan yaw from a sample of world normals
    Ns = np.concatenate([cap.nrm_cam[i][::7] @ cap.poses[i][:3, :3].T for i in frames[::3] if len(cap.nrm_cam[i])])
    M = rz(-manhattan_yaw(Ns))[:3, :3]
    raw = []   # (axis, sign, frame, t, m-array)
    for i in frames:
        if not len(cap.pts_cam[i]):
            continue
        R = M @ cap.poses[i][:3, :3]
        t = M @ cap.poses[i][:3, 3]
        Pw = cap.pts_cam[i] @ R.T          # camera-to-point vectors in the aligned world
        Nw = cap.nrm_cam[i] @ R.T
        for ax in range(3):
            for sg in (1, -1):
                sel = Nw[:, ax] * sg > cos_th
                if sel.sum() >= min_pts:
                    raw.append((ax, sg, i, t[ax], Pw[sel, ax]))
    # global planes per (axis, normal sign) group from all points at lambda = 1
    obs = []
    pid = 0
    for ax in range(3):
        for sg in (1, -1):
            grp = [r for r in raw if r[0] == ax and r[1] == sg]
            if not grp:
                continue
            v = np.concatenate([r[3] + r[4] for r in grp])
            edges = np.arange(v.min() - 0.02, v.max() + 0.03, 0.01)
            h = gaussian_filter1d(np.histogram(v, edges)[0].astype(float), 2.0)
            for pk in _peaks(h, max(500, 0.01 * len(v)), min_sep=15):
                c = edges[pk] + 0.005
                for _, _, i, t, m in grp:
                    near = np.abs(t + m - c) < win
                    if near.sum() >= min_pts:
                        obs.append((pid, t, float(np.median(m[near])), float(cap.timestamps[i])))
                pid += 1
    return np.array(obs, dtype=float).reshape(-1, 4)


def _solve(obs: np.ndarray, iters: int = 6, huber: float = 0.02) -> float:
    """lambda minimising sum_j sum_i w (t_i + lambda m_i - c_j)^2, IRLS with Huber weights."""
    pid, t, m = obs[:, 0].astype(int), obs[:, 1], obs[:, 2]
    w = np.ones(len(obs))
    lam = 1.0
    for _ in range(iters):
        num = den = 0.0
        tc, mc = np.empty_like(t), np.empty_like(m)
        for j in np.unique(pid):
            s = pid == j
            ws = w[s]
            if ws.sum() <= 0:
                tc[s] = mc[s] = 0
                continue
            tc[s] = t[s] - np.average(t[s], weights=ws)
            mc[s] = m[s] - np.average(m[s], weights=ws)
        num, den = -np.sum(w * tc * mc), np.sum(w * mc * mc)
        lam = num / den
        r = np.abs(tc + lam * mc)
        w = np.where(r < huber, 1.0, huber / np.maximum(r, 1e-9))
    return float(lam)


def estimate_depth_scale(cap, frame_step: int = 3, win: float = 0.10, min_pts: int = 80,
                         min_obs_per_plane: int = 8, block_s: float = 5.0, n_boot: int = 200,
                         seed: int = 0) -> DepthCalib | None:
    obs = _observations(cap, frame_step, win, min_pts)
    if len(obs) == 0:
        return None
    # keep planes seen often enough and from a spread of distances
    keep = np.zeros(len(obs), bool)
    spreads = []
    for j in np.unique(obs[:, 0]):
        s = obs[:, 0] == j
        sd = np.std(np.abs(obs[s, 2]))
        if s.sum() >= min_obs_per_plane and sd > 0.05:
            keep |= s
            spreads.append(sd)
    obs = obs[keep]
    if len(spreads) < 2:
        return None
    lam = _solve(obs)
    # block bootstrap over time chunks (pose drift is temporally correlated)
    rng = np.random.default_rng(seed)
    blk = np.floor((obs[:, 3] - obs[:, 3].min()) / block_s).astype(int)
    ub = np.unique(blk)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ub, len(ub), replace=True)
        idx = np.concatenate([np.where(blk == b)[0] for b in pick])
        boots.append(1.0 / _solve(obs[idx]))
    a = 1.0 / lam
    return DepthCalib(scale=a, se=float(np.std(boots)), n_planes=len(spreads), n_obs=len(obs),
                      range_spread_m=float(np.median(spreads)))


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from ..frontends.lidar import load_any

    for s in sys.argv[1:]:
        cap = load_any(Path(s))
        r = estimate_depth_scale(cap)
        if r is None:
            print(f"{Path(s).name}: not identifiable")
            continue
        print(f"{Path(s).name}: depth scale {r.scale:.4f} ({r.rel * 100:+.2f}% +- {r.se * 100:.2f}%)  "
              f"planes={r.n_planes} obs={r.n_obs} range spread={r.range_spread_m:.2f} m")


# ---------------------------------------------------------------------------------------------------
# per-device scale from a tape-measured reference (`roomscope calibrate-depth`)
# ---------------------------------------------------------------------------------------------------
REF_SD_M = 0.003       # a tape or laser distance meter read to the nearest few mm


def scale_from_references(result: dict, refs: dict[str, float], ref_sd: float = REF_SD_M) -> tuple[float, float, list[str]]:
    """a = reported / measured for each reference on an UNCORRECTED run. refs: {"R1:ceiling": 2.95,
    "R1-W3": 3.42, "ceiling": 2.95 (single room)}. Ceiling height scales fully with depth (floor and
    ceiling are both depth readings); a wall length only partly (wall positions also rest on VIO
    translation), so it under-estimates the scale offset; ceilings are preferred and labelled.
    Returns (a, se, notes). se combines the reference reading, the fit error, and the calibration
    device's capture-to-capture spread (one capture cannot show its own repeatability)."""
    rooms = {r["id"]: r for r in result["rooms"]}
    walls = {w["id"]: w for r in result["rooms"] for w in r["walls"]}
    a, var, notes = [], [], []
    for key, truth in refs.items():
        if key.endswith("ceiling"):
            rid = key.split(":")[0] if ":" in key else (next(iter(rooms)) if len(rooms) == 1 else None)
            if rid not in rooms:
                raise ValueError(f"{key}: name the room (e.g. R1:ceiling); rooms are {sorted(rooms)}")
            m = rooms[rid]["ceiling_height"]
            if m.get("observed") is False:
                raise ValueError(f"{key}: {rid}'s ceiling was not observed in this capture; scan the ceiling")
            kind = "ceiling"
        elif key in walls:
            m, kind = walls[key]["length"], "wall (partial scale sensitivity)"
        else:
            raise ValueError(f"{key}: no such room ceiling or wall id")
        v, s_fit = m["value"], (m["hi"] - m["value"]) / 1.645
        a.append(v / truth)
        var.append((v / truth) ** 2 * ((ref_sd / truth) ** 2 + (s_fit / v) ** 2))
        notes.append(f"{key} {kind}: reported {v:.4f} m / measured {truth:.4f} m = {v / truth:.4f}")
    w = 1 / np.asarray(var)
    est = float(np.sum(w * np.asarray(a)) / w.sum())
    se_meas = float(np.sqrt(1 / w.sum()))
    cfg = load_config()
    floor = float((cfg.get("devices") or {}).get(cfg.get("calibration_device", ""), {}).get("se", 0.0))
    spread = float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) >= 2 else 0.0
    return est, float(max(np.hypot(se_meas, floor), spread)), notes


def write_device_entry(device: str, scale: float, se: float, source: str, path=None) -> None:
    """Insert or replace one device block under `devices:` in config/depth_scale.yaml, keeping comments."""
    path = path or CONFIG
    lines = path.read_text().splitlines()
    block = [f'  "{device}":', f"    scale: {scale:.5f}", f"    se: {se:.5f}", f"    source: {json.dumps(source)}"]
    head = f'  "{device}":'
    if head in lines:
        i = lines.index(head)
        j = i + 1
        while j < len(lines) and lines[j].startswith("    "):
            j += 1
        lines[i:j] = block
    else:
        i = lines.index("devices:") + 1
        while i < len(lines) and lines[i].startswith("  ") and not lines[i].startswith("  #"):
            i += 1
        lines[i:i] = block
    path.write_text("\n".join(lines) + "\n")
