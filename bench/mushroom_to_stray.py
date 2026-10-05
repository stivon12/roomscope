"""Convert a MuSHRoom iPhone capture (Ren et al., arXiv 2311.02778, CC BY 4.0) into a Stray Scanner export,
so the unchanged LiDAR front-end reads it, and its Faro scan can score the result (bench/mushroom_eval.py).

    python bench/mushroom_to_stray.py benchmark/raw/mushroom/room_datasets/vr_room/iphone/long_capture \
        benchmark/raw/mushroom_stray/vr_room_long

Source (Polycam on an iPhone 12 Pro Max, checked on vr_room): images/frame_NNNNN.jpg 738x994 portrait,
depth/frame_NNNNN.png uint16 mm at the same size with no holes, transformations.json (nerfstudio): per-frame
fl_x, fl_y, cx, cy and transform_matrix = camera-to-world, OpenGL camera axes, z-up world (mean camera up
(0, -0.06, 0.95)). The Polycam poses are used, not transformations_colmap.json: they are what the phone
tracked, as in a walk-in capture; the COLMAP ones were refined offline over both captures.

What differs from a real Stray export, and how it is handled (all visible to the pipeline's output):
- depth was already upsampled and hole-filled by Polycam; it is resized (nearest) to 192 x 258, the
  portrait shape of ARKit's 256 x 192 LiDAR map, so the core sees the resolution it was tuned on;
- no confidence maps: every pixel is written as 2 (high), so the confidence filter does nothing;
- keyframes only (no timestamps, no IMU): timestamps are spaced 0.1 s so the 10 Hz thinning keeps all.
Pose: Stray stores camera-to-world with OpenCV camera axes in ARKit's y-up world, so
T_stray = ZUP_TO_ARKIT @ T_gl @ diag(1, -1, -1, 1).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from roomscope.frontends.lidar import ARKIT_TO_ZUP

DEPTH_WH = (192, 258)
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def convert(src: Path, dst: Path) -> int:
    meta = json.loads((src / "transformations.json").read_text())
    frames = sorted(meta["frames"], key=lambda f: f["file_path"])
    (dst / "depth").mkdir(parents=True, exist_ok=True)
    (dst / "confidence").mkdir(exist_ok=True)
    w, h = frames[0]["w"], frames[0]["h"]
    vid = cv2.VideoWriter(str(dst / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"]
    conf = np.full(DEPTH_WH[::-1], 2, np.uint8)
    for i, f in enumerate(frames):
        img = cv2.imread(str(src / f["file_path"]))
        assert img.shape[:2] == (h, w), f["file_path"]
        vid.write(img)
        d = cv2.imread(str(src / f["depth_file_path"]), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(dst / "depth" / f"{i:06d}.png"), cv2.resize(d, DEPTH_WH, interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(str(dst / "confidence" / f"{i:06d}.png"), conf)
        T = np.linalg.inv(ARKIT_TO_ZUP) @ np.asarray(f["transform_matrix"]) @ GL_TO_CV
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        x, y, z = T[:3, 3]
        rows.append(f"{0.1 * i:.3f}, {i:06d}, {x}, {y}, {z}, {q[0]}, {q[1]}, {q[2]}, {q[3]}, "
                    f"{f['fl_x']}, {f['fl_y']}, {f['cx']}, {f['cy']}, , ")
    vid.release()
    (dst / "odometry.csv").write_text("\n".join(rows) + "\n")
    f = frames[-1]
    np.savetxt(dst / "camera_matrix.csv", [[f["fl_x"], 0, f["cx"]], [0, f["fl_y"], f["cy"]], [0, 0, 1]],
               delimiter=",")
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    a = ap.parse_args()
    print(f"{convert(a.src, a.dst)} frames -> {a.dst}")


if __name__ == "__main__":
    main()
