"""D2/D3: how wrong are the ARKit poses, frame by frame? Depth-error-free.

Each `highres_depth` frame is the laser surface as seen from that frame. Back-projected with the
*ARKit* pose it should land exactly on the laser cloud; any per-frame misfit is ARKit pose error (plus
the ~1-2 cm ARKitScenes registration noise), with the phone LiDAR's own depth bias playing no part.

Per frame: point-to-plane ICP of the frame onto the laser (already registered into the result frame
by eval/laser.py) gives a correction dT_i. Frames that see too few independent plane directions are
flagged as unconstrained. Then:
- spread of dT_i over time = within-capture drift (H3);
- a similarity fit of ARKit camera centres to corrected centres = trajectory scale s (D2).

    python -m roomscope.eval.pose_drift <scene_dir> <out_dir_with_result.json+cloud.npz>
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from .laser import score_capture


def _frame_cloud(f: Path, scene: Path, traj: np.ndarray, T_res: np.ndarray, stride: int = 8):
    vid, tstr = f.stem.rsplit("_", 1)
    t = float(tstr)
    j = int(np.argmin(np.abs(traj[:, 0] - t)))
    if abs(traj[j, 0] - t) > 0.005:
        return None
    pin = None
    for c in (tstr, f"{t - 0.001:.3f}", f"{t + 0.001:.3f}"):
        p = scene / "lowres_wide_intrinsics" / f"{vid}_{c}.pincam"
        if p.exists():
            pin = np.loadtxt(p)
            break
    if pin is None:
        return None
    w, h, fx, fy, cx, cy = pin
    d = cv2.imread(str(f), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    H, W = d.shape
    sx, sy = W / w, H / h
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = d[v, u]
    ok = (z > 0.2) & (z < 5.0)
    Pc = np.c_[((u[ok] + 0.5) - cx * sx) / (fx * sx) * z[ok], ((v[ok] + 0.5) - cy * sy) / (fy * sy) * z[ok], z[ok]]
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = cv2.Rodrigues(traj[j, 1:4])[0], traj[j, 4:7]
    Tcw = np.linalg.inv(E)                       # camera(OpenCV) -> ARKit world
    T = np.eye(4); T[:3, :3] = T_res
    Tcw = T @ Tcw                                # -> result frame
    P = Pc @ Tcw[:3, :3].T + Tcw[:3, 3]
    return t, P, Tcw[:3, 3]


def run(scene: Path, out_dir: Path) -> str:
    scene, out_dir = Path(scene), Path(out_dir)
    sc, ref = score_capture(out_dir, scene, return_ref=True)   # laser registered into result frame
    T_res = np.load(out_dir / "cloud.npz")["T_world_to_result"]
    traj = np.loadtxt(scene / "lowres_wide.traj")
    ref = ref.voxel_down_sample(0.02)
    if not ref.has_normals():
        ref.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    rows = []
    for f in sorted((scene / "highres_depth").glob("*.png")):
        fc = _frame_cloud(f, scene, traj, T_res)
        if fc is None:
            continue
        t, P, c = fc
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        src = src.voxel_down_sample(0.03)
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
        reg = o3d.pipelines.registration.registration_icp(
            src, ref, 0.10, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        reg = o3d.pipelines.registration.registration_icp(
            src, ref, 0.04, reg.transformation, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        N = np.asarray(src.normals)
        ev = np.linalg.eigvalsh(N.T @ N / len(N))          # translation observability per direction
        dT = reg.transformation
        ang = np.degrees(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)))
        c_corr = dT[:3, :3] @ c + dT[:3, 3]
        rows.append((t, *(c_corr - c), ang, reg.fitness, reg.inlier_rmse, ev[0], *c, *c_corr))
    A = np.array(rows)
    good = (A[:, 5] > 0.6) & (A[:, 7] > 0.05)
    t0 = A[0, 0]
    lines = [f"scene {scene.name}: {len(A)} highres frames, {good.sum()} well-constrained (fitness>0.6, all 3 "
             f"directions observed)"]
    d = A[good, 1:4]
    lines.append("ARKit camera-position error vs laser (cm), after removing the capture-wide mean offset:")
    dm = d - np.median(d, 0)
    mag = np.linalg.norm(dm, axis=1)
    lines.append(f"  median |err| {np.median(mag) * 100:.1f}  p90 {np.percentile(mag, 90) * 100:.1f}  max "
                 f"{mag.max() * 100:.1f}  | per-axis std x {dm[:, 0].std() * 100:.1f} y {dm[:, 1].std() * 100:.1f} "
                 f"z {dm[:, 2].std() * 100:.1f}")
    lines.append(f"  rotation error: median {np.median(A[good, 4]):.2f} deg, max {A[good, 4].max():.2f} deg")
    lines.append("  error over time (10 equal time slices; median dx, dy, dz in cm):")
    tt = A[good, 0] - t0
    for k, (a, b) in enumerate(zip(np.linspace(0, tt.max(), 11)[:-1], np.linspace(0, tt.max(), 11)[1:])):
        m = (tt >= a) & (tt <= b)
        if m.sum():
            med = np.median(dm[m], 0) * 100
            lines.append(f"    t={a:5.0f}-{b:4.0f}s n={m.sum():3d}  dx {med[0]:+5.1f}  dy {med[1]:+5.1f}  dz {med[2]:+5.1f}")
    # D2: similarity fit ARKit centres -> corrected centres (Umeyama)
    X, Y = A[good, 9:12], A[good, 12:15]
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    s = np.trace(np.diag(S) @ D) / (Xc ** 2).sum(1).mean()
    lines.append(f"trajectory scale (similarity fit, {len(X)} centres, extent "
                 f"{np.ptp(X, 0).round(2).tolist()} m): s = {s:.4f} ({(s - 1) * 100:+.2f}%)")
    np.save(Path("out/diag") / f"pose_drift_{scene.name}.npy", A)
    return "\n".join(lines)


if __name__ == "__main__":
    Path("out/diag").mkdir(parents=True, exist_ok=True)
    txt = run(Path(sys.argv[1]), Path(sys.argv[2]))
    print(txt)
    (Path("out/diag") / f"pose_drift_{Path(sys.argv[1]).name}.txt").write_text(txt + "\n")
