"""Images -> metric posed points, for the video and photo tiers (no depth sensor, no poses).

Model: MapAnything (Meta, Apache-2.0 weights `facebook/map-anything-apache`, 1.23 B parameters,
4.9 GB). One feed-forward pass over all views returns, per view, metric camera-to-world poses,
intrinsics, per-pixel 3D points and confidence. Its output is turned into the same LidarCapture the
LiDAR loader produces, so the shared geometry core (planes, rooms, openings, drift, calibration)
runs unchanged.

Gravity: MapAnything's world frame is the first camera's frame, which says nothing about "up". Images
are upright (phone camera apps store orientation; ARKitScenes frames are rotated upright from their
sky_direction), so the cameras' average up direction is close to gravity. It is snapped to the
nearest of the three dominant surface-normal (Manhattan) axes so walls come out exactly vertical.

Weights are fetched by `scripts/fetch_weights.sh` into the Hugging Face cache, never into the repo.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np

from .lidar import LidarCapture, _normals_from_depth

MODEL_ID = os.environ.get("ROOMSCOPE_MAPANYTHING", "facebook/map-anything-apache")
MODEL_INFO = {"name": "MapAnything", "version": MODEL_ID, "license": "Apache-2.0",
              "use": "metric multi-view reconstruction (poses, intrinsics, points) for video/photo tiers"}
CV_TO_OURS = np.diag([1.0, -1.0, -1.0, 1.0])


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def image_intrinsics(path: Path) -> np.ndarray | None:
    """Pinhole K for an image at its stored (upright) resolution: a sidecar intrinsics.json written next
    to it (benchmark frames, video keyframes), else EXIF 35 mm-equivalent focal length. None if unknown,
    and MapAnything then estimates it (it under-estimated focal by 34% on ARKitScenes frames, which bends
    floors by ~20 deg, so a known K matters)."""
    import json
    side = Path(path).parent / "intrinsics.json"
    if side.exists():
        d = json.loads(side.read_text())
        if Path(path).name in d:
            return np.asarray(d[Path(path).name], float)
    from PIL import Image, ImageOps
    try:
        im = Image.open(path)
        f35 = im.getexif().get_ifd(0x8769).get(0xA405)       # FocalLengthIn35mmFilm
        W, H = ImageOps.exif_transpose(im).size
    except Exception:
        return None
    if not f35:
        return None
    f = float(f35) / 43.2666 * np.hypot(W, H)                 # 35 mm equivalent is defined on the diagonal
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def run_mapanything(image_paths: list[Path], conf_percentile: float = 30.0, cache: Path | None = None) -> list[dict]:
    """Returns per view: pts_cam (H,W,3) OpenCV camera frame, pose (4,4) cam->world (OpenCV), K (3,3),
    mask (H,W) bool. With `cache`, the raw model output is stored as .npz and reused when the same images
    are given again (deterministic replay, and no second 5 GB model load)."""
    import hashlib
    Ks = [image_intrinsics(p) for p in image_paths]
    use_K = all(k is not None for k in Ks)
    key = hashlib.sha1(("|".join(f"{p.name}:{p.stat().st_size}" for p in image_paths) + f"|K={use_K}").encode()).hexdigest()[:12]
    if cache is not None and (Path(cache) / f"mapanything_{key}.npz").exists():
        z = np.load(Path(cache) / f"mapanything_{key}.npz")
        return [{k: z[f"{k}_{i}"] for k in ("pts_cam", "pose", "K", "mask", "conf")} for i in range(int(z["n"]))]
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images, preprocess_inputs

    dev = _device()
    model = MapAnything.from_pretrained(MODEL_ID).to(dev).eval()
    if use_K:
        from PIL import Image, ImageOps
        raw = [{"img": np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB")),
                "intrinsics": torch.from_numpy(k.astype(np.float32))} for p, k in zip(image_paths, Ks)]
        views = preprocess_inputs(raw)
    else:
        views = load_images([str(p) for p in image_paths])
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=dev != "cpu",
                            amp_dtype="bf16" if dev == "cuda" else "fp16", apply_mask=True, mask_edges=True,
                            apply_confidence_mask=False)
    out = []
    for p in preds:
        out.append({
            "pts_cam": p["pts3d_cam"][0].float().cpu().numpy(),
            "pose": p["camera_poses"][0].float().cpu().numpy(),
            "K": p["intrinsics"][0].float().cpu().numpy(),
            "mask": p["mask"][0].squeeze(-1).bool().cpu().numpy(),
            "conf": p["conf"][0].float().cpu().numpy(),
        })
    # per-view confidence threshold: MapAnything's own percentile mask is computed across all views
    # and left 2-3% of pixels in some views of a 6-photo room
    for v in out:
        c = v["conf"].squeeze()
        v["mask"] = v["mask"] & (c >= np.percentile(c[v["mask"]], conf_percentile) if v["mask"].any() else False)
    del model, preds
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()
    if cache is not None:
        Path(cache).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(cache) / f"mapanything_{key}.npz", n=len(out),
                            **{f"{k}_{i}": v[k] for i, v in enumerate(out) for k in v})
    for v in out:
        v["K_source"] = np.array(1 if use_K else 0)
    return out


def _manhattan_up(normals: np.ndarray, cams: np.ndarray, up_prior: np.ndarray | None = None) -> np.ndarray:
    """Unit 'up' in the reconstruction's world frame (see module docstring)."""
    N = normals[np.random.default_rng(0).choice(len(normals), min(len(normals), 200_000), replace=False)]
    # dominant axis 1: principal eigenvector of the normals' scatter; then the orthogonal pair by
    # 4-theta circular mean in the plane perpendicular to it
    w, V = np.linalg.eigh(N.T @ N)
    axes = []
    a1 = V[:, -1]
    e1 = np.cross(a1, [1.0, 0, 0] if abs(a1[0]) < 0.9 else [0, 1.0, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(a1, e1)
    perp = N - np.outer(N @ a1, a1)
    keep = np.linalg.norm(perp, axis=1) > 0.9
    th = np.arctan2(perp[keep] @ e2, perp[keep] @ e1)
    phi = np.arctan2(np.mean(np.sin(4 * th)), np.mean(np.cos(4 * th))) / 4
    a2 = np.cos(phi) * e1 + np.sin(phi) * e2
    a3 = np.cross(a1, a2)
    axes = [a1, a2, a3]
    if up_prior is not None:
        # upright phone images: the cameras' mean up is gravity give or take the typical downward pitch
        # (up to ~30 deg). Refine on the floor/ceiling normals near it: principal direction of normals
        # within 35 deg of the current estimate, a few times. Snapping straight to a Manhattan axis from
        # the full normal scatter left a 27 deg tilt on 42444946.
        up = up_prior / np.linalg.norm(up_prior)
        for cone in (35, 25, 15, 10):
            sel = np.abs(N @ up) > np.cos(np.deg2rad(cone))
            if sel.sum() < 500:
                break
            w_, V_ = np.linalg.eigh(N[sel].T @ N[sel])
            up = V_[:, -1] * np.sign(V_[:, -1] @ up)
        return up
    c = cams - cams.mean(0)
    up = axes[int(np.argmin([np.var(c @ a) for a in axes]))]
    # sign: more surface facing up (floor) than facing down (ceiling)
    if (N @ up > 0.9).sum() < (N @ up < -0.9).sum():
        up = -up
    return up


def _rot_to_z(up: np.ndarray) -> np.ndarray:
    z = np.array([0, 0, 1.0])
    v, c = np.cross(up, z), float(up @ z)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def to_capture(views: list[dict], timestamps: np.ndarray, root: Path, pixel_stride: int = 2,
               scale: float = 1.0) -> LidarCapture:
    """MapAnything views -> LidarCapture (our camera frame x right, y up, -z forward; z-up world)."""
    pts, nrms, poses = [], [], []
    for v in views:
        P = v["pts_cam"] * scale                         # OpenCV camera frame
        Pours = P * np.array([1.0, -1.0, -1.0])          # -> x right, y up, -z forward
        valid = v["mask"] & (P[..., 2] > 0.1)
        n, nok = _normals_from_depth(Pours, valid)
        keep = (valid & nok)[::pixel_stride, ::pixel_stride]
        pts.append(Pours[::pixel_stride, ::pixel_stride][keep].astype(np.float32))
        nrms.append(n[::pixel_stride, ::pixel_stride][keep].astype(np.float32))
        T = v["pose"].astype(np.float64).copy()
        T[:3, 3] *= scale
        poses.append(T @ CV_TO_OURS)
    poses = np.asarray(poses)
    Nw = np.concatenate([n @ T[:3, :3].T for n, T in zip(nrms, poses)])
    # images are upright (phone camera apps record orientation; benchmark frames are rotated upright),
    # so each camera's up is its +y in our convention (-y in OpenCV)
    up_prior = poses[:, :3, 1].mean(0)
    up = _manhattan_up(Nw, poses[:, :3, 3], up_prior / np.linalg.norm(up_prior))
    G = np.eye(4)
    G[:3, :3] = _rot_to_z(up)
    poses = np.einsum("ij,fjk->fik", G, poses)
    K = views[0]["K"].astype(np.float64)
    return LidarCapture(pts, nrms, poses, K, np.asarray(timestamps, float), Path(root), align=G)



def sfm_poses(image_paths: list[Path], work: Path, K: np.ndarray | None, sequential: bool) -> dict[str, np.ndarray]:
    """Classical SfM (COLMAP via pycolmap, CPU): SIFT, matching, incremental mapping with bundle adjustment.
    Returns {image name: 4x4 camera->world (OpenCV), up to an unknown global scale} for the largest
    reconstruction. MapAnything's poses are not globally consistent enough to fuse (per-view scale +-10%,
    floors smeared over ~0.9 m on 42444946); bundle-adjusted SfM poses are."""
    import shutil

    import pycolmap
    work = Path(work)
    shutil.rmtree(work, ignore_errors=True)
    img_dir = work / "images"
    img_dir.mkdir(parents=True)
    for p in image_paths:
        shutil.copy(p, img_dir / p.name)
    db = work / "database.db"
    ro = pycolmap.ImageReaderOptions()
    if K is not None:
        ro.camera_model = "PINHOLE"
        ro.camera_params = f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}"
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro,
                              device=pycolmap.Device.cpu)
    if sequential:
        po = pycolmap.SequentialPairingOptions()
        po.overlap = 12
        pycolmap.match_sequential(db, pairing_options=po, device=pycolmap.Device.cpu)
    else:
        pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
    opts = pycolmap.IncrementalPipelineOptions()
    if K is not None:
        opts.mapper.abs_pose_refine_focal_length = False
        opts.mapper.abs_pose_refine_extra_params = False
        opts.ba_refine_focal_length = False
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params = False
    recs = pycolmap.incremental_mapping(db, img_dir, work / "sparse", options=opts)
    if not recs:
        return {}
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    out = {}
    for img in rec.images.values():
        if not (img.has_pose() if callable(img.has_pose) else img.has_pose):
            continue
        cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
        M = np.eye(4)
        M[:3, :4] = cfw.matrix()
        Twc = np.linalg.inv(M)
        pts = [(p.xy, rec.points3D[p.point3D_id].xyz) for p in img.points2D if p.has_point3D()]
        out[img.name] = {"pose": Twc, "obs": pts}
    return out


def fuse_sfm_depth(views: list[dict], names: list[str], sfm: dict, image_sizes: list[tuple[int, int]],
                   K_known: list[np.ndarray | None]) -> tuple[list[dict], list[int], dict]:
    """SfM poses + MapAnything dense depth.
    - rays: known intrinsics when given (MapAnything largely ignores the intrinsics input), else the
      model's own rays;
    - per-view depth scale k_i = median(SfM depth / MapAnything depth) over the view's triangulated
      SIFT points, which removes MapAnything's per-view scale scatter;
    - global metric scale: median of MapAnything / SfM depth over all those points (MapAnything is
      metric; on 42444946 its depth was 4% short of LiDAR). Calibration then measures what remains.
    Returns the fused views (OpenCV camera frame, metric), the indices kept, and diagnostics."""
    out, kept, ratios = [], [], []
    per_view = {}
    for i, (v, nm) in enumerate(zip(views, names)):
        if nm not in sfm:
            continue
        H, W = v["pts_cam"].shape[:2]
        W0, H0 = image_sizes[i]
        sc = W / W0                                     # resize to model width, centre crop in height
        oy = (H0 * sc - H) / 2
        d_ma = np.linalg.norm(v["pts_cam"], axis=-1)    # depth along ray
        T = sfm[nm]["pose"]
        Tinv = np.linalg.inv(T)
        zs, zm = [], []
        for xy, X in sfm[nm]["obs"]:
            u, vv = xy[0] * sc, xy[1] * sc - oy
            iu, iv = int(u), int(vv)
            if not (0 <= iu < W and 0 <= iv < H) or not v["mask"][iv, iu]:
                continue
            Xc = Tinv[:3, :3] @ X + Tinv[:3, 3]
            if Xc[2] <= 0:
                continue
            zs.append(np.linalg.norm(Xc)); zm.append(d_ma[iv, iu])
        if len(zs) < 20:
            continue
        r = np.array(zs) / np.array(zm)
        k = float(np.median(r))
        per_view[nm] = (k, len(zs), float(np.median(np.abs(r / k - 1))))
        ratios += list(np.array(zm) / np.array(zs))
        if K_known[i] is not None:
            Kk = K_known[i].copy(); Kk[:2] *= sc; Kk[1, 2] -= oy
        else:
            Kk = v["K"]
        uu, vg = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        ray = np.stack([(uu - Kk[0, 2]) / Kk[0, 0], (vg - Kk[1, 2]) / Kk[1, 1], np.ones_like(uu)], -1)
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        out.append({"pts_cam": ray * (d_ma * k)[..., None], "pose": T.copy(), "K": Kk, "mask": v["mask"],
                    "conf": v.get("conf")})
        kept.append(i)
    if not out:
        return [], [], {"registered": 0}
    s = float(np.median(ratios))                         # SfM units -> metres
    for o in out:
        o["pts_cam"] = o["pts_cam"] * s
        o["pose"][:3, 3] *= s
    diag = {"registered": len(out), "of": len(views), "metric_scale_from_mapanything": s,
            "per_view_scale_spread": float(np.std([np.log(k) for k, _, _ in per_view.values()])),
            "median_view_residual": float(np.median([e for _, _, e in per_view.values()]))}
    return out, kept, diag
