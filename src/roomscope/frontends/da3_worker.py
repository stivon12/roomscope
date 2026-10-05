"""Depth Anything 3 (ByteDance, Apache-2.0 checkpoints only), run in its OWN venv.

DA3's package imports pycolmap (pycolmap + torch abort Python on macOS: duplicate libomp) and a set of
export/UI libraries we never call, so it cannot share the main environment. It lives in .venv-da3 and
is called as a subprocess by frontends/recon.py and bench/scale_cues.py:

    .venv-da3/bin/python -m roomscope.frontends.da3_worker <job.json> <out.npz>

job.json:
  {"mode": "mono_metric" | "multiview" | "posed",
   "images": [path, ...],
   "K": [3x3 | null, ...]            pinhole K of each image at its stored (upright) resolution,
   "poses_c2w": [4x4, ...]           posed only: OpenCV camera->world, any scale (COLMAP),
   "metric": bool                    multiview/posed: also run DA3METRIC-LARGE for a metric scale}

Modes (models: DA3-BASE 0.12 B and DA3METRIC-LARGE 0.35 B, both Apache-2.0):
- mono_metric: DA3METRIC-LARGE per image. Metric depth = focal_px_at_processing_res * output / 300
  (DA3 README, "Monocular Metric Depth").
- multiview:  DA3-BASE on all images jointly: depth, confidence, poses (up to scale), intrinsics.
- posed:      DA3-BASE conditioned on the given K and poses; depth is returned in the poses' units
  (align_to_input_ext_scale=True) and the output poses are the inputs.

out.npz per view i: depth_<i> (z-depth at processing resolution), conf_<i>, pose_<i> (4x4 c2w OpenCV),
K_<i> (3x3 at processing resolution), size_<i> (orig W, H); with "metric": metric_<i> (DA3METRIC
z-depth resized to depth_<i>'s grid). Images are resized (long side 504, each side to a multiple of
14), never cropped, so K scales by w/W0 and h/H0.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

# imported by DA3 modules we never use (COLMAP/GLB/gaussian export, web UI); stubbed so that importing
# the API does not need them. pycolmap must never be real here (libomp clash with torch).
_UNUSED = {"pycolmap", "gradio", "fastapi", "uvicorn", "typer", "pydantic", "gsplat", "xformers",
           "open3d", "sklearn"}


class _Stub(types.ModuleType):
    __path__: list = []

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return type(k, (), {"__init__": lambda s, *a, **kw: None})


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in _UNUSED:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        return _Stub(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _StubFinder())

BASE_ID = "depth-anything/DA3-BASE"
METRIC_ID = "depth-anything/DA3METRIC-LARGE"
PROCESS_RES = 504


def _load(paths):
    from PIL import Image, ImageOps
    return [np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p in paths]


def _model(mid, dev):
    from depth_anything_3.api import DepthAnything3
    return DepthAnything3.from_pretrained(mid).to(dev).eval()


def _free(model, dev):
    import gc
    import torch
    del model
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()


def _scaled_K(K, W0, H0, w, h):
    K = np.asarray(K, np.float64).copy()
    K[0] *= w / W0
    K[1] *= h / H0
    return K


def _metric(imgs, Ks, dev):
    """DA3METRIC-LARGE per image -> metric z-depth list. Focal from the known K, else 0.8 x long side."""
    import torch
    model = _model(METRIC_ID, dev)
    out = []
    for im, K in zip(imgs, Ks):
        with torch.no_grad():
            pred = model.inference([im], process_res=PROCESS_RES)
        d = np.asarray(pred.depth[0], np.float32)
        h, w = d.shape
        H0, W0 = im.shape[:2]
        f = (K[0][0] * w / W0 + K[1][1] * h / H0) / 2 if K is not None else 0.8 * max(w, h)
        out.append(d * f / 300.0)
    _free(model, dev)
    return out


def main(job_path: str, out_path: str):
    import cv2
    import torch

    job = json.loads(Path(job_path).read_text())
    mode, paths = job["mode"], job["images"]
    Ks = job.get("K") or [None] * len(paths)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    imgs = _load(paths)
    t0 = time.time()
    out = {"n": len(paths)}
    if mode == "mono_metric":
        for i, (im, d) in enumerate(zip(imgs, _metric(imgs, Ks, dev))):
            out[f"depth_{i}"], out[f"size_{i}"] = d, np.array([im.shape[1], im.shape[0]])
    else:
        known = all(k is not None for k in Ks)
        kw = {}
        if known:
            kw["intrinsics"] = np.asarray(Ks, np.float32)
        if mode == "posed":
            kw["extrinsics"] = np.linalg.inv(np.asarray(job["poses_c2w"], np.float64)).astype(np.float32)
            kw["align_to_input_ext_scale"] = True
        model = _model(BASE_ID, dev)
        with torch.no_grad():
            pred = model.inference(imgs, process_res=PROCESS_RES, **kw)
        _free(model, dev)
        for i, im in enumerate(imgs):
            d = np.asarray(pred.depth[i], np.float32)
            h, w = d.shape
            H0, W0 = im.shape[:2]
            E = np.eye(4)
            E[:3, :4] = np.asarray(pred.extrinsics[i])[:3, :4]
            out[f"depth_{i}"] = d
            out[f"conf_{i}"] = np.asarray(pred.conf[i], np.float32)
            out[f"pose_{i}"] = np.linalg.inv(E)
            out[f"K_{i}"] = _scaled_K(Ks[i], W0, H0, w, h) if known else np.asarray(pred.intrinsics[i], np.float64)
            out[f"size_{i}"] = np.array([W0, H0])
        if job.get("metric"):
            for i, m in enumerate(_metric(imgs, Ks, dev)):
                h, w = out[f"depth_{i}"].shape
                out[f"metric_{i}"] = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    out["seconds"] = time.time() - t0
    np.savez_compressed(out_path, **out)
    print(f"da3 {mode}: {len(paths)} images in {out['seconds']:.1f}s on {dev}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
