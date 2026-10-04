# roomscope

Phone capture → dimensioned floor plan with an interval on every number.
One command per capture, three capture tiers sharing one output contract (`schema/plan.schema.json`):

| Tier | Input | Status (measured against Faro laser scans, ARKitScenes) |
|---|---|---|
| **LiDAR** | Stray Scanner export, or an ARKitScenes raw scene | Works. Ceiling within 1 cm on 3/3 held-out captures; walls with both neighbours observed: median length error 1.4 cm; calibrated intervals (leave-one-room-out coverage 0.94). |
| **Photo** | one folder of 2–8 photos per room | Runs end to end. Wall lengths +4 % / +1 % / −3 % on three rooms; **ceiling height unreliable** (eye-level photos rarely show the ceiling); not yet calibrated. |
| **Video** | one walkthrough video | Runs end to end; **not accurate yet** (MapAnything scale and pose consistency; see `docs/RESEARCH.md` §4, §7). |

What is measured and what is not is tracked in `docs/COMPLIANCE.md`; the fix loop is in `fix/`.

## Install (macOS Apple Silicon or Linux, Python 3.11)

```bash
git clone <this repo> roomscope && cd roomscope
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[models,dev]"
```

LiDAR tier needs nothing else. Photo and video tiers also need MapAnything (Apache-2.0 weights):

```bash
git clone https://github.com/facebookresearch/map-anything.git third_party/map-anything
git -C third_party/map-anything checkout 3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9
uv pip install --python .venv/bin/python -e third_party/map-anything
scripts/fetch_weights.sh          # ~4.9 GB into ~/.cache/huggingface, never into the repo
```

Optional, video SfM experiment only: `uv pip install --python .venv/bin/python pycolmap`.
Do not import `pycolmap` and `torch` in the same Python process on macOS (duplicate OpenMP runtime
aborts Python); the pipeline runs COLMAP in a subprocess for that reason.

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
src/roomscope/frontends/   lidar.py  video.py  photo.py  recon.py (MapAnything)  sfm_worker.py (COLMAP)
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
| MapAnything (`facebook/map-anything-apache`, 1.23 B) | photo and video depth + poses | Apache-2.0 |
| COLMAP / pycolmap (SIFT only) | video SfM experiment (opt-in) | BSD |

Every result records the models it used in `meta.models`.
