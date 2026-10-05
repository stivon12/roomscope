# roomscope

Phone capture → dimensioned floor plan with an interval on every number.
One command per capture, three capture tiers sharing one output contract (`schema/plan.schema.json`):

| Tier | Input | Status (measured against Faro laser scans, ARKitScenes) |
|---|---|---|
| **LiDAR** | Stray Scanner export, or an ARKitScenes raw scene | Runs end to end. Ceiling within 1 cm on 3/3 held-out captures; walls: against the structural laser reference, median wall-length error 8.7 cm on the 21 cleanly referenced walls, 9 of them off by more than 10 cm (the 1.4 cm quoted earlier came from a scorer that matched our own edges; see docs/WALL_ERRORS.md). Interval calibration must be refit against the new reference. |
| **Photo** | one folder of 2–8 photos per room | Runs end to end and stitches per-room folders (YC apartment: 5 of 6 rooms placed). **Not accurate yet:** metric scale off by 3–12 %, so ceilings are 7–28 cm off; the ±8 % wall gate is not met. |
| **Video** | one walkthrough video | Runs end to end (COLMAP poses + Depth Anything 3 depth). **Not accurate yet:** metric scale off by 3–12 % (ceilings 8–28 cm); the ±3 % wall gate is not met. On multi-room walks COLMAP splits the walk into pieces and only the largest is kept. |

Per-tier accuracy is in `docs/DEVICE_MATRIX.md`, gates on the benchmark set in `bench/results/gates.md`, the report in `docs/REPORT.md`, the compliance matrix in `docs/COMPLIANCE.md` and the fix loop in `fix/`.

## Install (macOS Apple Silicon or Linux, Python 3.11+, one environment)

Needs [uv](https://docs.astral.sh/uv/) (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone <this repo> roomscope && cd roomscope
uv sync --extra video --extra dev     # one .venv for every tier; LiDAR only: uv sync --extra dev
scripts/fetch_weights.sh              # ~7.9 GB into ~/.cache/huggingface (video/photo tiers, labels, damage)
.venv/bin/roomscope doctor            # checks packages, GPU and weights; says which tiers are ready
```

Measured on an M1 MacBook (16 GB) with empty caches (2026-10-05, before the damage models were added): `uv sync` 37 s, weights 182 s (2.9 GB), `pytest` 40 s: **4 min 20 s** in total. The damage models add ~5 GB of download (not yet re-timed), and `pytest` now runs the full pipeline including damage (7 min on this machine). `.venv` is 1.4 GB. Download time scales with your connection.

Why there is only one environment although two of its packages clash: `pycolmap` and `torch` each ship
an OpenMP runtime, and loading both in one Python process aborts it on macOS. They are never loaded
together: COLMAP (`frontends/sfm_worker.py`) and Depth Anything 3 (`frontends/da3_worker.py`) run as
subprocesses of the same interpreter. Depth Anything 3 is installed from a pinned commit; its declared
dependencies include `xformers` (no macOS build), `numpy<2` and a web UI, which `[tool.uv]` in
`pyproject.toml` drops; the worker stubs the modules it never calls.

The old MapAnything path (`ROOMSCOPE_RECON=mapanything`) is optional: clone and install
`third_party/map-anything` at commit `3d10cf7a` and run `scripts/fetch_weights.sh --mapanything` (+4.9 GB).

## Capture

Follow `docs/PROTOCOL.md` (one page): stock Camera app for photos and video, Stray Scanner for LiDAR.

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

Ablation switches (for the report only, not for normal use): `--no-drift`, `--no-depth-correction`,
`--depth-scale A` (override the LiDAR depth scale, e.g. from one tape-measured distance), `--device NAME`.

Typical run time on an M1 Pro (16 GB): LiDAR 1–3 min; photo ~1 min per room; video 5–7 min
(peak memory ~10–12 GB: run one capture at a time).

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
docs/                      PROTOCOL.md  COMPLIANCE.md  DIAGNOSTICS.md  RESEARCH.md
fix/                       DECLARATION.md  before_after.md
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
