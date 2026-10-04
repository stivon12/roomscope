"""Known-answer test for the Stray Scanner loader.

No real Stray export is in the repo, so one is synthesised from an ARKitScenes capture exactly as Stray
writes it (strayrobots/scanner: docs/format.md, OdometryEncoder.swift):
- odometry.csv header with spaces after commas, per-frame fx, fy, cx, cy for the RGB image;
- pose = ARKit camera-to-world (y-up world) times q_AC, a 180 deg rotation about x, i.e. OpenCV camera axes;
- depth/NNNNNN.png uint16 mm, confidence/NNNNNN.png 0/1/2; depth at 256x192, RGB 1920x1440.
Loading it with load_stray must give the same world points as load_arkitscenes for the same frames.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from roomscope.frontends.lidar import ARKIT_TO_ZUP, load_arkitscenes, load_stray

SCENE = Path(__file__).resolve().parents[1] / "benchmark/raw/arkitscenes/raw/Validation/42444946"
pytestmark = pytest.mark.skipif(not (SCENE / "lowres_depth").is_dir(), reason="ARKitScenes 42444946 not downloaded")


def _write_stray(dst: Path, n: int = 25) -> list[float]:
    depths = sorted((SCENE / "lowres_depth").glob("*.png"))[::40][:n]
    traj = np.loadtxt(SCENE / "lowres_wide.traj")
    (dst / "depth").mkdir(parents=True)
    (dst / "confidence").mkdir()
    rows, kept = [], []
    for k, dp in enumerate(depths):
        t = float(dp.stem.split("_")[1])
        j = int(np.argmin(np.abs(traj[:, 0] - t)))
        if abs(traj[j, 0] - t) > 0.005:
            continue
        E = np.eye(4)
        E[:3, :3], E[:3, 3] = cv2.Rodrigues(traj[j, 1:4])[0], traj[j, 4:7]
        T_cv_zup = np.linalg.inv(E)                          # camera(OpenCV) -> world (z up)
        T_stray = np.linalg.inv(ARKIT_TO_ZUP) @ T_cv_zup     # Stray: ARKit y-up world, OpenCV camera axes
        q = Rotation.from_matrix(T_stray[:3, :3]).as_quat()  # x, y, z, w
        pin = SCENE / "lowres_wide_intrinsics" / f"{dp.stem}.pincam"
        w, h, fx, fy, cx, cy = np.loadtxt(pin)
        s = 1920 / w                                         # Stray intrinsics are for the RGB image
        fr = len(kept)
        cv2.imwrite(str(dst / "depth" / f"{fr:06d}.png"), cv2.imread(str(dp), cv2.IMREAD_UNCHANGED))
        cv2.imwrite(str(dst / "confidence" / f"{fr:06d}.png"),
                    cv2.imread(str(SCENE / "confidence" / dp.name), cv2.IMREAD_UNCHANGED))
        x, y, z = T_stray[:3, 3]
        rows.append(f"{t}, {fr}, {x}, {y}, {z}, {q[0]}, {q[1]}, {q[2]}, {q[3]}, "
                    f"{fx * s}, {fy * s}, {cx * s}, {cy * s}, 0, 0")
        kept.append(t)
    (dst / "odometry.csv").write_text("timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, "
                                      "distortion_center_x, distortion_center_y\n" + "\n".join(rows) + "\n")
    return kept


def test_stray_matches_arkitscenes(tmp_path):
    ts = _write_stray(tmp_path / "rec")
    stray = load_stray(tmp_path, hz=1000.0)                  # keep every synthesised frame
    ark = load_arkitscenes(SCENE)
    Ginv = np.linalg.inv(ark.align)                          # ARKitScenes loader also gravity-aligns
    assert len(stray.poses) == len(ts)
    errs = []
    for i, t in enumerate(stray.timestamps):
        j = int(np.argmin(np.abs(ark.timestamps - t)))
        assert abs(ark.timestamps[j] - t) < 1e-3
        Pa = ark.pts_cam[j] @ (Ginv @ ark.poses[j])[:3, :3].T + (Ginv @ ark.poses[j])[:3, 3]
        Ps, _, _ = stray.world(i)
        m = min(len(Pa), len(Ps))
        errs.append(np.median(np.linalg.norm(Pa[:m] - Ps[:m], axis=1)))
    assert np.median(errs) < 0.005, f"Stray and ARKitScenes clouds differ by {np.median(errs):.3f} m"
    assert stray.load_warnings and "rgb.mp4" in stray.load_warnings[0]
