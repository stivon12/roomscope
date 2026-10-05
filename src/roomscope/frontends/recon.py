"""Images -> metric posed points, for the video and photo tiers (no depth sensor, no poses).

Default model: Depth Anything 3 (ByteDance; Apache-2.0 checkpoints DA3-BASE, 0.12 B parameters, and
DA3METRIC-LARGE, 0.35 B), run in its own venv by frontends/da3_worker.py:
- photos: DA3-BASE on all views of a room jointly -> depth, confidence, poses (up to scale), intrinsics;
- video:  DA3-BASE conditioned on COLMAP's K and poses -> depth in COLMAP units, consistent with them.
DA3-BASE is not metric; metres come from DA3METRIC-LARGE's monocular depth on the same views (median
ratio). bench/scale_cues.py measures that cue against LiDAR.
MapAnything (Meta, Apache-2.0 `facebook/map-anything-apache`, 1.23 B) is kept behind
ROOMSCOPE_RECON=mapanything for comparison. Either output becomes the same LidarCapture the LiDAR
loader produces, so the shared geometry core (planes, rooms, openings, drift, calibration) runs unchanged.

Gravity: the reconstruction's world frame is the first camera's frame, which says nothing about "up". Images
are upright (phone camera apps store orientation; ARKitScenes frames are rotated upright from their
sky_direction), so the cameras' average up direction is close to gravity. It is snapped to the
nearest of the three dominant surface-normal (Manhattan) axes so walls come out exactly vertical.

Weights are fetched into the Hugging Face cache, never into the repo.
"""
from __future__ import annotations

import gc
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

from .lidar import LidarCapture, _normals_from_depth

RECON = os.environ.get("ROOMSCOPE_RECON", "da3")
MODEL_ID = os.environ.get("ROOMSCOPE_MAPANYTHING", "facebook/map-anything-apache")
MODEL_INFO = ({"name": "Depth Anything 3", "version": "depth-anything/DA3-BASE + depth-anything/DA3METRIC-LARGE",
               "license": "Apache-2.0",
               "use": "multi-view / pose-conditioned depth (BASE) and monocular metric scale (METRIC-LARGE)"}
              if RECON == "da3" else
              {"name": "MapAnything", "version": MODEL_ID, "license": "Apache-2.0",
               "use": "metric multi-view reconstruction (poses, intrinsics, points) for video/photo tiers"})
CV_TO_OURS = np.diag([1.0, -1.0, -1.0, 1.0])
ROOT = Path(__file__).resolve().parents[3]


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


def reray_known_K(views: list[dict], image_paths: list[Path]) -> int:
    """Replace MapAnything's predicted ray directions with the KNOWN camera's, keeping its depth along each
    ray. Given true intrinsics, MapAnything still predicts its own rays: on the 42444946 photos its focal
    ranged 0.68-1.09x the true one per view, so each view's surfaces were unprojected through a different
    wrong field of view (floors tilted 25-60 deg, vertical span 4 m in a 3 m room). Images are resized to
    the model width and centre-cropped (MapAnything fixed-mapping preprocessing). Returns views changed."""
    from PIL import Image
    n = 0
    for v, p in zip(views, image_paths):
        K = image_intrinsics(p)
        if K is None:
            continue
        from PIL import ImageOps
        W0, H0 = ImageOps.exif_transpose(Image.open(p)).size   # the model sees the upright image
        H, W = v["pts_cam"].shape[:2]
        sc = max(W / W0, H / H0)
        ox, oy = (W0 * sc - W) / 2, (H0 * sc - H) / 2
        Kk = K.astype(np.float64).copy()
        Kk[:2] *= sc
        Kk[0, 2] -= ox
        Kk[1, 2] -= oy
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        ray = np.stack([(uu - Kk[0, 2]) / Kk[0, 0], (vv - Kk[1, 2]) / Kk[1, 1], np.ones_like(uu)], -1)
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        dist = np.linalg.norm(v["pts_cam"], axis=-1, keepdims=True)
        v["pts_cam"] = (ray * dist).astype(np.float32)
        v["K_model"], v["K"] = v["K"], Kk.astype(np.float32)
        n += 1
    return n


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


def _rz(a: float) -> np.ndarray:
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])


def manhattan_snap(poses: np.ndarray, nrms: list[np.ndarray], min_pts: int = 300,
                   max_tilt_deg: float = 15.0) -> tuple[np.ndarray, list[dict]]:
    """Per-view rotation correction under the Manhattan-world assumption (gravity-aligned, z up).

    Feed-forward reconstructions get individual view rotations wrong by a few degrees, and one view in
    eight by ~18 deg (42444946 photos vs ARKit, MapAnything and VGGT alike), which is enough to break the
    room layout. Interiors have one vertical and two horizontal dominant directions, so each view is
    rotated about its own camera centre:
    1. tilt: its floor/ceiling normals onto +z (if it sees >= min_pts of them, correction <= max_tilt);
    2. yaw: its wall normals onto the room's dominant axes, the nearest one (correction within +-45 deg).
    The room's axes come from all views, iteratively ignoring views more than 10 deg off."""
    from ..core.layout import manhattan_yaw
    poses = poses.copy()
    wrap = lambda a: (a + np.pi / 4) % (np.pi / 2) - np.pi / 4
    log = []
    for i in range(len(poses)):                       # tilt
        Nw = nrms[i] @ poses[i, :3, :3].T
        h = Nw[np.abs(Nw[:, 2]) > 0.85]
        if len(h) >= min_pts:
            u = (h * np.sign(h[:, 2])[:, None]).mean(0)
            u /= np.linalg.norm(u)
            ang = np.degrees(np.arccos(np.clip(u[2], -1, 1)))
            if ang <= max_tilt_deg:
                poses[i, :3, :3] = _rot_to_z(u) @ poses[i, :3, :3]
                log.append({"view": i, "tilt_deg": round(float(ang), 2)})
    yaws, wts = np.zeros(len(poses)), np.zeros(len(poses))
    for i in range(len(poses)):
        Nw = nrms[i] @ poses[i, :3, :3].T
        wall = np.abs(Nw[:, 2]) < 0.3
        if wall.sum() >= min_pts:
            yaws[i], wts[i] = manhattan_yaw(Nw), wall.sum()
    ok = wts > 0
    if ok.sum() == 0:
        return poses, log
    g = 0.0
    for _ in range(3):
        c, s_ = np.average(np.cos(4 * yaws[ok]), weights=wts[ok]), np.average(np.sin(4 * yaws[ok]), weights=wts[ok])
        g = np.arctan2(s_, c) / 4
        ok = (wts > 0) & (np.abs(wrap(yaws - g)) < np.deg2rad(10)) if ((wts > 0) & (np.abs(wrap(yaws - g)) < np.deg2rad(10))).sum() >= 2 else ok
    for i in np.where(wts > 0)[0]:
        d = wrap(g - yaws[i])
        poses[i, :3, :3] = _rz(d) @ poses[i, :3, :3]
        log.append({"view": int(i), "yaw_deg": round(float(np.degrees(d)), 2)})
    return poses, log


def to_capture(views: list[dict], timestamps: np.ndarray, root: Path, pixel_stride: int = 2,
               scale: float = 1.0, snap: bool = True) -> LidarCapture:
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
    snap_log = []
    if snap:
        poses, snap_log = manhattan_snap(poses, nrms)
    K = views[0]["K"].astype(np.float64)
    cap = LidarCapture(pts, nrms, poses, K, np.asarray(timestamps, float), Path(root), align=G)
    cap.snap_log = snap_log
    return cap



def sfm_poses(image_paths: list[Path], work: Path, K: np.ndarray | None, overlap: int = 15) -> dict:
    """Classical SfM (COLMAP, SIFT, sequential matching, fixed known pinhole camera) on dense frames.
    Runs in a subprocess (frontends/sfm_worker.py): pycolmap and torch cannot share a process on macOS.
    Returns {image name: {"pose": 4x4 camera->world (OpenCV, arbitrary scale), "obs": [(xy, XYZ), ...]}}
    for the largest reconstruction. MapAnything's own poses are not consistent enough to fuse (floors
    smeared over ~0.9 m on 42444946); bundle-adjusted SfM poses are."""
    import json
    import shutil
    import subprocess
    import sys

    work = Path(work)
    img_dir = work / "images"
    same = (work / "sfm.json").exists() and img_dir.is_dir() and \
        sorted((q.name, q.stat().st_size) for q in img_dir.iterdir()) == sorted((q.name, q.stat().st_size) for q in image_paths)
    if same and (work / "sfm.args.json").exists() and json.loads((work / "sfm.args.json").read_text()) == \
            {"overlap": overlap, "K": None if K is None else np.round(K, 3).tolist()}:
        print("SfM: reusing", work / "sfm.json")
        raw = json.loads((work / "sfm.json").read_text())
        return {nm: {"pose": np.asarray(r["pose"]), "obs": [(np.array(o[:2]), np.array(o[2:])) for o in r["obs"]]}
                for nm, r in raw.items()}
    shutil.rmtree(img_dir, ignore_errors=True)
    img_dir.mkdir(parents=True)
    for p in image_paths:
        shutil.copy(p, img_dir / p.name)
    cmd = [sys.executable, "-m", "roomscope.frontends.sfm_worker", str(img_dir), str(work), "--overlap", str(overlap)]
    if K is not None:
        cmd += ["--K", *(f"{v:.4f}" for v in (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))]
    subprocess.run(cmd, check=True)
    (work / "sfm.args.json").write_text(json.dumps({"overlap": overlap, "K": None if K is None else np.round(K, 3).tolist()}))
    raw = json.loads((work / "sfm.json").read_text())
    return {nm: {"pose": np.asarray(r["pose"]), "obs": [(np.array(o[:2]), np.array(o[2:])) for o in r["obs"]]}
            for nm, r in raw.items()}


def clean_sfm_track(sfm: dict, times: dict[str, float], k: int = 3, factor: float = 4.0) -> tuple[dict, list[str]]:
    """Drop registered frames that break the continuity of a handheld camera path. A walking camera cannot
    jump metres between frames ~0.5 s apart, but COLMAP occasionally registers a frame into a wrong place
    (42444946, exhaustive matching: 13 of 145 frames 1.3-2.4 m off ARKit while the rest were 1.9 cm).
    A frame is dropped when its distance to the median of its +-k time neighbours exceeds `factor` times
    the median such distance. Returns (kept sfm, dropped names)."""
    names = sorted((n for n in sfm if n in times), key=lambda n: times[n])
    if len(names) < 2 * k + 3:
        return sfm, []
    C = np.array([np.asarray(sfm[n]["pose"])[:3, 3] for n in names])
    dev = np.array([np.linalg.norm(C[i] - np.median(np.delete(C[max(0, i - k):i + k + 1], min(i, k), 0), 0))
                    for i in range(len(C))])
    thr = factor * np.median(dev)
    drop = [n for n, d in zip(names, dev) if d > thr]
    return {n: v for n, v in sfm.items() if n not in drop}, drop


def run_mapanything_posed(image_paths: list[Path], K: np.ndarray, poses_c2w: list[np.ndarray],
                          cache: Path | None = None, conf_percentile: float = 30.0) -> tuple[list[dict], float]:
    """MapAnything conditioned on known intrinsics AND camera poses (OpenCV cam2world, arbitrary scale), the
    recipe of the repo's scripts/demo_inference_on_colmap_outputs.py. The paper reports ~5 % metric-scale
    error with images + intrinsics + poses vs ~13 % from images + intrinsics alone (arXiv 2509.13414,
    Table 2). Returns per-view dicts (pts_cam re-projected through the known K at model resolution, metric
    pose from the model) and the metric scale factor applied to the input poses."""
    import hashlib
    key = hashlib.sha1(("posed|" + "|".join(f"{p.name}:{p.stat().st_size}" for p in image_paths)
                        + "|" + np.array2string(np.round(K, 3)) + "|"
                        + "|".join(np.array2string(np.round(T, 4)) for T in poses_c2w)).encode()).hexdigest()[:12]
    cp = None if cache is None else Path(cache) / f"mapanything_posed_{key}.npz"
    if cp is not None and cp.exists():
        z = np.load(cp)
        out = [{k: z[f"{k}_{i}"] for k in ("pts_cam", "pose", "K", "mask", "conf")} for i in range(int(z["n"]))]
        return out, float(z["scale"])
    import torch
    from PIL import Image, ImageOps
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    dev = _device()
    model = MapAnything.from_pretrained(MODEL_ID).to(dev).eval()
    raw = [{"img": np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB")),
            "intrinsics": torch.from_numpy(K.astype(np.float32)),
            "camera_poses": torch.from_numpy(np.asarray(T, np.float32)),
            "is_metric_scale": torch.tensor([False])} for p, T in zip(image_paths, poses_c2w)]
    views = preprocess_inputs(raw)
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, minibatch_size=1,
                            ignore_calibration_inputs=False, ignore_depth_inputs=True, ignore_pose_inputs=False,
                            ignore_depth_scale_inputs=True, ignore_pose_scale_inputs=True,
                            use_amp=dev != "cpu", amp_dtype="bf16" if dev == "cuda" else "fp16",
                            apply_mask=True, mask_edges=True, apply_confidence_mask=False)
    out = []
    for p, pr in zip(image_paths, preds):
        dz = pr["depth_z"][0].squeeze(-1).float().cpu().numpy()
        H, W = dz.shape
        W0, H0 = ImageOps.exif_transpose(Image.open(p)).size
        sc = max(W / W0, H / H0)
        Kk = K.astype(np.float64).copy()
        Kk[:2] *= sc
        Kk[0, 2] -= (W0 * sc - W) / 2
        Kk[1, 2] -= (H0 * sc - H) / 2
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        pts = np.stack([(uu - Kk[0, 2]) / Kk[0, 0] * dz, (vv - Kk[1, 2]) / Kk[1, 1] * dz, dz], -1)
        out.append({"pts_cam": pts.astype(np.float32), "pose": pr["camera_poses"][0].float().cpu().numpy(),
                    "K": Kk.astype(np.float32), "mask": pr["mask"][0].squeeze(-1).bool().cpu().numpy(),
                    "conf": pr["conf"][0].float().cpu().numpy()})
    for v in out:
        c = v["conf"].squeeze()
        v["mask"] = v["mask"] & (c >= np.percentile(c[v["mask"]], conf_percentile)) if v["mask"].any() else v["mask"]
    # metric scale the model put on the input poses: ratio of camera-to-camera distances, output / input
    Ci = np.array([np.asarray(T)[:3, 3] for T in poses_c2w])
    Co = np.array([v["pose"][:3, 3] for v in out])
    iu = np.triu_indices(len(Ci), 1)
    di, do = np.linalg.norm(Ci[iu[0]] - Ci[iu[1]], axis=1), np.linalg.norm(Co[iu[0]] - Co[iu[1]], axis=1)
    good = di > np.percentile(di, 25)
    scale = float(np.median(do[good] / di[good]))
    del model, preds
    gc.collect()
    if dev == "mps":
        torch.mps.empty_cache()
    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, n=len(out), scale=scale, **{f"{k}_{i}": v[k] for i, v in enumerate(out) for k in v})
    return out, scale


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


# ---------------------------------------------------------------- Depth Anything 3 (default)

def _da3_job(job: dict, cache: Path | None, tag: str) -> dict:
    """Run frontends/da3_worker.py in a subprocess of this interpreter (cached by image names/sizes and job inputs)."""
    import hashlib
    import json
    import subprocess
    import tempfile
    key = hashlib.sha1((tag + json.dumps(job, sort_keys=True) + "|".join(
        f"{Path(p).name}:{Path(p).stat().st_size}" for p in job["images"])).encode()).hexdigest()[:12]
    cp = None if cache is None else Path(cache) / f"da3_{tag}_{key}.npz"
    if cp is not None and cp.exists():
        return dict(np.load(cp))
    py = sys.executable
    if importlib.util.find_spec("depth_anything_3") is None:
        raise RuntimeError("Depth Anything 3 is not installed; see README, Install (video and photo tiers)")
    with tempfile.TemporaryDirectory() as td:
        jp, op = Path(td) / "job.json", Path(td) / "out.npz"
        jp.write_text(json.dumps(job))
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        subprocess.run([str(py), "-m", "roomscope.frontends.da3_worker", str(jp), str(op)], check=True, env=env)
        z = dict(np.load(op))
    if cp is not None:
        cp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cp, **z)
    return z


def _da3_views(z: dict, conf_percentile: float) -> list[dict]:
    """Worker output -> view dicts (pts_cam OpenCV from z-depth and K at processing resolution)."""
    out = []
    for i in range(int(z["n"])):
        dz, K = z[f"depth_{i}"].astype(np.float64), z[f"K_{i}"].astype(np.float64)
        H, W = dz.shape
        uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
        pts = np.stack([(uu - K[0, 2]) / K[0, 0] * dz, (vv - K[1, 2]) / K[1, 1] * dz, dz], -1)
        c = z[f"conf_{i}"]
        m = np.isfinite(dz) & (dz > 0)
        if m.any():
            m &= c >= np.percentile(c[m], conf_percentile)
        out.append({"pts_cam": pts.astype(np.float32), "pose": z[f"pose_{i}"].astype(np.float64),
                    "K": K.astype(np.float32), "mask": m, "conf": c,
                    "metric_z": z.get(f"metric_{i}")})
    return out


def _metric_scale(views: list[dict]) -> tuple[float, list[float]]:
    """Metres per reconstruction unit: per view the median of DA3METRIC z / reconstruction z over its
    confident pixels, then the median over views."""
    per = []
    for v in views:
        mz, z = v.get("metric_z"), v["pts_cam"][..., 2]
        if mz is None:
            continue
        m = v["mask"] & (mz > 0)
        if m.sum() > 500:
            per.append(float(np.median(mz[m] / z[m])))
    return (float(np.median(per)) if per else 1.0), per


def run_da3(image_paths: list[Path], cache: Path | None = None, conf_percentile: float = 30.0) -> list[dict]:
    """Photos: DA3-BASE multi-view (known K passed when every image has one), scaled to metres by
    DA3METRIC-LARGE. Returns view dicts like run_mapanything (pts_cam, pose, K, mask, conf)."""
    Ks = [image_intrinsics(p) for p in image_paths]
    use_K = all(k is not None for k in Ks)
    job = {"mode": "multiview", "images": [str(p) for p in image_paths],
           "K": [k.tolist() for k in Ks] if use_K else None, "metric": True}
    views = _da3_views(_da3_job(job, cache, "mv"), conf_percentile)
    s, per = _metric_scale(views)
    for v in views:
        v["pts_cam"] = v["pts_cam"] * s
        v["pose"][:3, 3] *= s
        v["K_source"] = np.array(1 if use_K else 0)
    run_da3.diag = {"metric_scale": s, "per_view_metric_scale": per}
    return views


def run_da3_posed(image_paths: list[Path], K: np.ndarray, sfm: dict, cache: Path | None = None,
                  conf_percentile: float = 30.0) -> tuple[list[dict], float, dict]:
    """Video: DA3-BASE conditioned on COLMAP's K and poses, depth in COLMAP units. Each view's depth is
    then corrected by k_i = median(SfM z / DA3 z) over its triangulated SIFT points (DA3 got single views
    25 % off on 42444946), and everything is put in metres by DA3METRIC-LARGE (_metric_scale).
    Returns views (metric, OpenCV), metres per SfM unit, diagnostics."""
    poses = [np.asarray(sfm[p.name]["pose"], float) for p in image_paths]
    job = {"mode": "posed", "images": [str(p) for p in image_paths], "K": [np.asarray(K).tolist()] * len(image_paths),
           "poses_c2w": [T.tolist() for T in poses], "metric": True}
    z = _da3_job(job, cache, "posed")
    views = _da3_views(z, conf_percentile)
    ks, npts = [], []
    for v, p, T in zip(views, image_paths, poses):
        dz = v["pts_cam"][..., 2]
        H, W = dz.shape
        W0, H0 = z[f"size_{len(ks)}"]
        Tinv = np.linalg.inv(T)
        r = []
        for xy, X in sfm[p.name]["obs"]:
            Xc = Tinv[:3, :3] @ X + Tinv[:3, 3]
            u, w = int(xy[0] * W / W0), int(xy[1] * H / H0)
            if Xc[2] > 0 and 0 <= u < W and 0 <= w < H and v["mask"][w, u]:
                r.append(Xc[2] / dz[w, u])
        ks.append(float(np.median(r)) if len(r) >= 20 else np.nan)
        npts.append(len(r))
    kmed = float(np.nanmedian(ks)) if np.isfinite(ks).any() else 1.0
    ks = [k if np.isfinite(k) else kmed for k in ks]
    for v, k in zip(views, ks):
        v["pts_cam"] = v["pts_cam"] * k
    s, per = _metric_scale(views)
    for v in views:
        v["pts_cam"] = v["pts_cam"] * s
        v["pose"][:3, 3] *= s
    diag = {"recon": "da3", "da3_seconds": float(z["seconds"]), "per_view_sfm_correction": [round(k, 4) for k in ks],
            "sfm_points_per_view": npts, "per_view_metric_scale": [round(x, 4) for x in per]}
    return views, s, diag


def reconstruct(image_paths: list[Path], cache: Path | None = None) -> list[dict]:
    """Unposed images -> metric views, with the backend chosen by ROOMSCOPE_RECON."""
    if RECON == "mapanything":
        views = run_mapanything(image_paths, cache=cache)
        reray_known_K(views, image_paths)       # MapAnything ignores the given K; DA3 uses it directly
        return views
    return run_da3(image_paths, cache=cache)


def reconstruct_posed(image_paths: list[Path], K: np.ndarray, sfm: dict,
                      cache: Path | None = None) -> tuple[list[dict], float, dict]:
    """COLMAP-posed images -> metric views, metres per SfM unit, diagnostics (ROOMSCOPE_RECON backend)."""
    if RECON == "mapanything":
        views, s = run_mapanything_posed(image_paths, K, [sfm[p.name]["pose"] for p in image_paths], cache=cache)
        return views, s, {"recon": "mapanything"}
    return run_da3_posed(image_paths, K, sfm, cache=cache)
