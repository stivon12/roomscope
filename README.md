# roomscope

Phone capture → dimensioned multi-room floor plan, with a calibrated interval on every number.
One command per capture (`roomscope run <capture>`), three capture tiers, one output contract (`schema/plan.schema.json`).
Every accuracy number below is scored against Faro laser scans (ARKitScenes, MuSHRoom).

| Tier | Input | Status |
|---|---|---|
| **LiDAR** | Stray Scanner export, or an ARKitScenes raw scene | Runs end to end, with damage regions, concealed-damage flags and scope items. **Ceiling:** within 1 cm on 3/3 held-out captures; iPad rooms within 0.3–0.8 cm on the benchmark, repeat spread 0.5–0.6 cm; the MuSHRoom iPhone reads ceilings about 2.4 % short (honka −2.7 / −3.3 cm, Polycam −3.1 cm), so the ceiling gate is met on 2/4 rooms. **Walls:** median wall-length error 8.7 cm on 21 walls scored against an independent structural laser reference (9 over 10 cm). That reference replaced an earlier scorer that matched our own edges and reported 1.4 cm (`docs/WALL_ERRORS.md`). Intervals refit on rooms disjoint from the benchmark (ceiling about ±3.5 cm). Depth scale measured per device (`config/depth_scale.yaml`). |
| **Photo** | one folder of 2–8 photos per room | Runs end to end on per-room folders; all 6 YC apartment rooms reconstructed. Calibrated intervals cover the laser value. **Known limits:** metric scale off by 3–12 % (ceilings 7–28 cm off), so the ±8 % wall gate is not met; no doors are detected at this tier, so 1 of 6 rooms is connected and the whole-property stitch is not met yet. |
| **Video** | one walkthrough video | Runs end to end (COLMAP poses + Depth Anything 3 depth). Calibrated intervals cover the laser value. **Known limits:** metric scale off by 3–12 % (ceilings 8–28 cm off), so the ±3 % wall gate is not met; on multi-room walks COLMAP splits the walk and the largest piece is kept. |

**Highlights**
- **Independent evaluation:** a structural laser reference, independent of where our edges sit, exposed and corrected optimistic wall numbers.
- **Calibration at every tier:** Fix 3 took video/photo interval coverage from 0/7 to 7/7, meeting its declared prediction.
- **Drift correction** recovers a bathroom that merges into the hallway without it (report §3).
- **Damage, concealed-damage flags and scope items** from rule tables with public citations (EPA, BRE, 40 CFR 745).
- **Head-to-head vs Polycam:** beat or tie on 7 of 8 dimensions (88 %, gate 70 %: met); every shared wall position within 1.2 cm of the laser.
- **Fix 4:** LiDAR repeatability 21 % → 40 %, honka pair 4/6 within 1 cm, all declared predictions met.

**[Technical report](TECHNICAL_REPORT.md)** ([PDF](TECHNICAL_REPORT.pdf)): architecture, tiers and device matrix, drift, error budget, calibration, fix loop, failure modes.

**[Fix declaration](fix/DECLARATION.md)** ([PDF](fix/DECLARATION.pdf)): each fix loop's worst gate, root cause with evidence, predicted number and outcome.

**[Capture protocol](CAPTURE_PROTOCOL.md)** ([PDF](CAPTURE_PROTOCOL.pdf)): the one-page guide for capturing a space at each tier.

Per-tier accuracy is in `docs/DEVICE_MATRIX.md`, gates on the benchmark set in `bench/results/gates.md`, the compliance matrix in `docs/COMPLIANCE.md` and the fix loop in `fix/`.

## Install (macOS Apple Silicon or Linux, Python 3.11+, one environment)

Needs [uv](https://docs.astral.sh/uv/) (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone <this repo> roomscope && cd roomscope
uv sync --extra video --extra dev     # one .venv for every tier; LiDAR only: uv sync --extra dev
scripts/fetch_weights.sh              # ~7.9 GB into ~/.cache/huggingface (video/photo tiers, labels, damage)
.venv/bin/roomscope doctor            # checks packages, GPU and weights; says which tiers are ready
```

Fresh install on an M1 MacBook (16 GB), empty caches, 2026-10-05: `uv sync` 37 s, weights 182 s (2.9 GB), `pytest` 40 s,
**4 min 20 s** in total (measured before the damage models were added; they add ~5 GB of download, not yet
re-timed, and `pytest` including damage takes 7 min). `.venv` is 1.4 GB; download time scales with your connection.

**One environment, two OpenMP runtimes.** `pycolmap` and `torch` each ship an OpenMP runtime, and loading both
in one Python process aborts it on macOS. COLMAP (`frontends/sfm_worker.py`) and Depth Anything 3
(`frontends/da3_worker.py`) therefore run as subprocesses of the same interpreter. Depth Anything 3 is installed
from a pinned commit; `[tool.uv]` in `pyproject.toml` drops its unneeded dependencies (`xformers`, which has no
macOS build, `numpy<2` and a web UI), and the worker stubs the modules it never calls.

The old MapAnything path (`ROOMSCOPE_RECON=mapanything`) is optional: clone and install
`third_party/map-anything` at commit `3d10cf7a` and run `scripts/fetch_weights.sh --mapanything` (+4.9 GB).

## Capture

Follow the [capture protocol](CAPTURE_PROTOCOL.md) ([PDF](CAPTURE_PROTOCOL.pdf), one page): stock Camera app for photos and video, Stray Scanner for LiDAR.

## Run

```bash
.venv/bin/roomscope run <capture>            # tier auto-detected; --tier photo|video|lidar to force
.venv/bin/roomscope validate out/<capture>/result.json
```

Output: `out/<capture>/result.json` (schema-valid), `out/<capture>/plan.png`, `out/<capture>/cloud.npz`.

| Capture folder contains | Tier |
|---|---|
| `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv` (Stray), or ARKitScenes `lowres_depth/` | LiDAR |
| a video file (`.mov`, `.mp4`) | video |
| one sub-folder of photos per room (or a single folder = one room) | photo |

Ablation switches (for the report, not for normal use): `--no-drift`, `--no-depth-correction`,
`--depth-scale A` (override the LiDAR depth scale, e.g. from one tape-measured distance), `--device NAME`.

Typical run time on an M1 Pro (16 GB): LiDAR 1–3 min; photo ~1 min per room; video 5–7 min
(peak memory ~10–12 GB; run one capture at a time).

## Reproduce the benchmark numbers

```bash
# ground truth: ARKitScenes raw scenes + Faro laser scans for the captures in bench/scenes.yaml
git clone https://github.com/apple/ARKitScenes.git third_party/ARKitScenes   # tested at 474aaa8
python third_party/ARKitScenes/download_data.py raw --split Validation --video_id <ids...> \
    --download_dir benchmark/raw/arkitscenes --download_laser_scanner_point_cloud \
    --raw_dataset_assets lowres_depth confidence lowres_wide.traj lowres_wide_intrinsics highres_depth mov
python bench/calibrate.py --tier lidar          # runs + scores every capture, fits config/calibration.json
python bench/fix_loop.py                        # fix 1 before/after  -> fix/before_after.md
python bench/wall_audit.py                      # fix 2 root-cause evidence
python bench/drift_eval.py <scene> <no-drift-out>   # drift ablation vs laser
python bench/drift_inject.py <scene>            # drift known-answer test
pytest -q tests/
```

## Layout

```
src/roomscope/frontends/   lidar.py  video.py  photo.py  recon.py  da3_worker.py (Depth Anything 3)  sfm_worker.py (COLMAP)
src/roomscope/core/        layout.py (planes, rooms, polygons)  drift.py  depth_calib.py  calibrate.py  stitch.py
src/roomscope/eval/        laser.py (Faro scoring)  depth_bias.py  pose_drift.py
bench/                     benchmark, calibration, ablation and fix-loop scripts
config/                    depth_scale.yaml  calibration.json
docs/                      COMPLIANCE.md  DEVICE_MATRIX.md  DIAGNOSTICS.md  RESEARCH.md
fix/                       DECLARATION.md (+ .pdf)  before_after.md
```

Weights, datasets and run outputs are never committed (`.gitignore`); folder layout is kept with `.gitkeep`.

## Models and licences

| Model | Used for | Licence |
|---|---|---|
| Depth Anything 3 (`depth-anything/DA3-BASE` 0.12 B, `DA3METRIC-LARGE` 0.35 B) | photo and video depth, metric scale | Apache-2.0 |
| COLMAP / pycolmap (SIFT only) | video camera poses | BSD |
| EoMT-L (`tue-mps/ade20k_semantic_eomt_large_512`) | indoor semantic labels (in progress) | MIT |
| MapAnything (`facebook/map-anything-apache`, 1.23 B) | old photo/video path, opt-in | Apache-2.0 |

Every result records the models it used in `meta.models`.
