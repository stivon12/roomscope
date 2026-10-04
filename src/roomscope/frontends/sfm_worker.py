"""COLMAP SfM in its own process (never import torch here).

pycolmap and torch each ship a libomp; loading both in one process aborts Python ("OMP: Error #15",
hloc issue #491, docs/RESEARCH.md section 7), so frontends/recon.py runs this module as a subprocess.

    python -m roomscope.frontends.sfm_worker <image_dir> <work_dir> [--K fx fy cx cy] [--overlap 15]

Writes <work_dir>/sfm.json: {image name: {"pose": 4x4 camera->world (OpenCV, arbitrary scale),
"obs": [[u, v, X, Y, Z], ...]}} for the largest reconstruction.
SIFT only: the pip pycolmap build has no ONNX, so ALIKED / LightGlue abort the process.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def run(image_dir: Path, work: Path, K: list[float] | None, overlap: int = 15, mapper: str = "incremental",
        init_tri_angle: float = 16.0, matching: str = "auto") -> dict:
    import pycolmap
    work = Path(work)
    shutil.rmtree(work / "sparse", ignore_errors=True)
    db = work / "database.db"
    db.unlink(missing_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    ro = pycolmap.ImageReaderOptions()
    if K is not None:                       # one shared, fixed pinhole camera
        ro.camera_model = "PINHOLE"
        ro.camera_params = ",".join(f"{v:.4f}" for v in K)
    eo = pycolmap.FeatureExtractionOptions()
    eo.max_image_size = 1024
    eo.sift.max_num_features = 8192
    # indoor walls are low-texture: default SIFT found ~700 keypoints/frame (some frames 0) on 42444946.
    # Lower peak threshold + domain-size pooling (COLMAP FAQ "increase number of matches")
    eo.sift.peak_threshold = 0.002
    eo.sift.domain_size_pooling = True
    eo.sift.estimate_affine_shape = True
    pycolmap.extract_features(db, image_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro,
                              extraction_options=eo, device=pycolmap.Device.cpu)
    v = pycolmap.TwoViewGeometryOptions()
    for name, val in (("use_degensac", True), ("compute_relative_pose", True), ("detect_watermark", False)):
        if hasattr(v, name):          # DEGENSAC (4.2+) is meant for plane-dominated scenes
            setattr(v, name, val)
    n_img = sum(1 for f in Path(image_dir).iterdir() if f.suffix.lower() in (".jpg", ".png"))
    if matching == "exhaustive" or (matching == "auto" and n_img <= 300):
        # all pairs: a room walk revisits walls, and those revisits are the loop closures that keep scale
        # from drifting. Sequential quadratic matching found 0 matches beyond a 16-frame gap on 42444946
        pycolmap.match_exhaustive(db, verification_options=v, device=pycolmap.Device.cpu)
    else:
        po = pycolmap.SequentialPairingOptions()
        po.overlap = overlap
        po.quadratic_overlap = False
        pycolmap.match_sequential(db, pairing_options=po, verification_options=v, device=pycolmap.Device.cpu)
    if mapper == "global":           # GLOMAP: rotation averaging first, robust to small baselines
        g = pycolmap.GlobalPipelineOptions()
        if K is not None:
            for name in ("refine_focal_length", "refine_principal_point", "refine_extra_params"):
                if hasattr(g.mapper.bundle_adjustment, name):
                    setattr(g.mapper.bundle_adjustment, name, False)
        recs = pycolmap.global_mapping(db, image_dir, work / "sparse", options=g)
    else:
        opts = pycolmap.IncrementalPipelineOptions()
        opts.mapper.init_min_tri_angle = init_tri_angle   # default 16 deg; a handheld room scan turns more than it moves
        if hasattr(opts, "structure_less_registration_fallback"):
            # COLMAP 4.0 default True: registers images from 2D-2D relative pose without 3D points; with weak
            # translation (planar/panoramic pairs) it stacked all cameras on one centre on 42444946
            opts.structure_less_registration_fallback = False
        opts.mapper.abs_pose_refine_focal_length = False
        opts.mapper.abs_pose_refine_extra_params = False
        if K is not None:
            opts.ba_refine_focal_length = False
            opts.ba_refine_principal_point = False
            opts.ba_refine_extra_params = False
        recs = pycolmap.incremental_mapping(db, image_dir, work / "sparse", options=opts)
    if not recs:
        return {}
    sizes = sorted((r.num_reg_images() for r in recs.values()), reverse=True)
    print(f"models: {len(recs)}, sizes {sizes}")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    out = {}
    for img in rec.images.values():
        if not (img.has_pose() if callable(img.has_pose) else img.has_pose):
            continue
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        M = np.eye(4)
        M[:3, :4] = cfw.matrix()
        obs = [[float(p.xy[0]), float(p.xy[1]), *map(float, rec.points3D[p.point3D_id].xyz)]
               for p in img.points2D if p.has_point3D()]
        out[img.name] = {"pose": np.linalg.inv(M).tolist(), "obs": obs}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("work", type=Path)
    ap.add_argument("--K", type=float, nargs=4, default=None)
    ap.add_argument("--overlap", type=int, default=15)
    ap.add_argument("--mapper", default="incremental", choices=["incremental", "global"])
    ap.add_argument("--matching", default="auto", choices=["auto", "exhaustive", "sequential"])
    ap.add_argument("--init-tri-angle", type=float, default=16.0)
    a = ap.parse_args()
    res = run(a.image_dir, a.work, a.K, a.overlap, a.mapper, a.init_tri_angle, a.matching)
    a.work.mkdir(parents=True, exist_ok=True)
    (a.work / "sfm.json").write_text(json.dumps(res))
    print(f"registered {len(res)} images")


if __name__ == "__main__":
    main()
