"""Photo tier: 2-8 stills per room, one sub-folder per room, no depth and no poses.

Each room folder is reconstructed on its own by MapAnything (frontends/recon.py) and analysed by the
shared core as a single room. The rooms are then placed into one property frame by core/stitch.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .recon import run_mapanything, to_capture

IMG_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def room_folders(path: Path) -> list[Path]:
    path = Path(path)
    dirs = sorted(d for d in path.iterdir() if d.is_dir() and any(f.suffix.lower() in IMG_EXT for f in d.iterdir()))
    if not dirs and any(f.suffix.lower() in IMG_EXT for f in path.iterdir()):
        dirs = [path]                            # a single folder of photos = one room
    return dirs


def images(folder: Path) -> list[Path]:
    ims = sorted(f for f in folder.iterdir() if f.suffix.lower() in IMG_EXT)
    if any(f.suffix.lower() in (".heic", ".heif") for f in ims):
        from PIL import Image
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError as e:
            raise RuntimeError("HEIC photos need `pip install pillow-heif` (or export as JPEG)") from e
        conv = []
        for f in ims:
            if f.suffix.lower() in (".heic", ".heif"):
                j = f.with_suffix(".jpg")
                if not j.exists():
                    Image.open(f).convert("RGB").save(j, quality=95)
                conv.append(j)
            else:
                conv.append(f)
        ims = conv
    return ims


def load_room(folder: Path, scale: float = 1.0):
    ims = images(folder)
    if len(ims) < 2:
        raise ValueError(f"{folder.name}: need at least 2 photos, found {len(ims)}")
    views = run_mapanything(ims, cache=Path(folder) / ".cache")
    return to_capture(views, np.arange(len(ims), dtype=float), folder, scale=scale)
