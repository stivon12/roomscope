"""Video tier evaluation on ARKitScenes captures (with .mov): camera poses vs ARKit, metric scale, laser.

    python bench/video_eval.py <scene_dir> [<scene_dir> ...] [--out out/video_eval] [--rerun]

Per capture:
- SfM: frames registered / dropped as path jumps; Sim3 ATE of the registered frames vs ARKit's trajectory
  (robust: inliers re-fitted below 2.5x median error; inlier share reported). Frame time = the frame's
  .pincam timestamp (lowres_wide.traj starts later than the video).
- Metric scale: the reconstruction's metric camera centres vs ARKit (both metric): scale of the similarity fit,
  reported as error in %. This is the number the +-3 % video gate depends on. ARKit's trajectory is VIO,
  itself good to ~1-2 cm here, so it is a reference, not truth.
- Laser: ceiling height error, wall-plane offsets, wall lengths (eval/laser.py, rigid registration).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

from roomscope.eval.laser import score_capture
from roomscope.pipeline import run_capture


def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rr = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / Xc.var(0).sum()
    return s, Rr, my - s * Rr @ mx


def robust_sim3(X, Y, iters=5):
    m = np.ones(len(X), bool)
    for _ in range(iters):
        s, Rr, t = umeyama(X[m], Y[m])
        E = np.linalg.norm(s * X @ Rr.T + t - Y, axis=1)
        m = E < max(2.5 * np.median(E[m]), 0.05)
    return s, E, m


def arkit_centres(scene: Path, times: np.ndarray):
    T = np.loadtxt(scene / "lowres_wide.traj")
    C = -np.einsum("nji,nj->ni", R.from_rotvec(T[:, 1:4]).as_matrix(), T[:, 4:7])
    ok = (times >= T[0, 0]) & (times <= T[-1, 0])
    return ok, interp1d(T[:, 0], C, axis=0)(times[ok])


def evaluate(scene: Path, out: Path, rerun: bool) -> dict:
    res_dir = out / scene.name
    if rerun or not (res_dir / "result.json").exists():
        run_capture(scene, "video", out, calibrate=False)
    row = {"capture": scene.name}
    sfm_p, diag_p = res_dir / "sfm" / "sfm.json", res_dir / "video_diag.json"
    if diag_p.exists():
        diag = json.loads(diag_p.read_text())
        row.update(registered=diag["registered"], dense=diag["dense"], dropped=diag["dropped_jumps"],
                   metric_scale_on_sfm=round(diag["metric_scale"], 4),
                   recon=diag.get("recon", "mapanything"), da3_seconds=diag.get("da3_seconds"))
        ft = diag["frame_times"]
        sfm = json.loads(sfm_p.read_text())
        names = sorted(n for n in sfm if n in ft and n not in set(diag["dropped"]))
        ok, Cg = arkit_centres(scene, np.array([ft[n] for n in names]))
        Cs = np.array([np.asarray(sfm[n]["pose"])[:3, 3] for n in names])[ok]
        _, E, m = robust_sim3(Cs, Cg)
        row.update(sfm_ate_median_cm=round(float(np.median(E[m])) * 100, 1),
                   sfm_ate_p90_cm=round(float(np.percentile(E[m], 90)) * 100, 1), sfm_inlier_share=round(float(m.mean()), 2))
        pk = diag["picked"]
        ok2, Cg2 = arkit_centres(scene, np.array([ft[n] for n in pk]))
        Cm = np.array(list(pk.values()))[ok2]
        s, E2, m2 = robust_sim3(Cm, Cg2)
        row.update(metric_scale_err_pct=round((1 / s - 1) * 100, 2), metric_views_inliers=f"{m2.sum()}/{len(m2)}",
                   metric_ate_median_cm=round(float(np.median(E2[m2])) * 100, 1))
    else:
        row["note"] = "SfM path not used (fell back to unposed reconstruction)"
    try:
        sc = score_capture(res_dir, scene)
        row["ceiling_err_cm"] = [round(c["err"] * 100, 1) for c in sc["ceil"]]
        pe = [abs(w["offset_err"]) * 100 for w in sc["wall_planes"]]
        le = [abs(w["err"]) * 100 for w in sc["walls"]]
        rl = [abs(w["rel"]) * 100 for w in sc["walls"]]
        row.update(wall_plane_median_cm=round(float(np.median(pe)), 1) if pe else None,
                   wall_len_median_cm=round(float(np.median(le)), 1) if le else None,
                   wall_len_median_pct=round(float(np.median(rl)), 1) if rl else None,
                   walls_scored=len(le), walls_reported=sum(len(r["walls"]) for r in json.loads((res_dir / "result.json").read_text())["rooms"]))
    except Exception as e:
        row["laser"] = f"{type(e).__name__}: {e}"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/video_eval"))
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    rows = []
    for sc in a.scenes:
        try:
            rows.append(evaluate(sc, a.out, a.rerun))
        except Exception as e:
            rows.append({"capture": sc.name, "error": f"{type(e).__name__}: {e}"})
        print(json.dumps(rows[-1]), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
