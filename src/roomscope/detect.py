"""Infer the input tier from the layout of a capture directory or file."""
from __future__ import annotations

from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".heic", ".png"}
VIDEO_EXT = {".mov", ".mp4", ".m4v"}


def detect_tier(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() in VIDEO_EXT:
            return "video"
        raise ValueError(f"unrecognised capture file: {path}")
    # Stray Scanner export: odometry.csv + depth/ (possibly one level down)
    if any(path.glob("odometry.csv")) or any(path.glob("*/odometry.csv")):
        return "lidar"
    if any(p.suffix.lower() in VIDEO_EXT for p in path.iterdir()):
        return "video"
    room_dirs = [d for d in path.iterdir() if d.is_dir()]
    if room_dirs and all(any(f.suffix.lower() in IMAGE_EXT for f in d.iterdir()) for d in room_dirs):
        return "photo"
    if any(f.suffix.lower() in IMAGE_EXT for f in path.iterdir()):
        return "photo"  # single room given as a flat folder
    raise ValueError(f"cannot infer tier for {path}; expected Stray export, a video, or per-room photo folders")
