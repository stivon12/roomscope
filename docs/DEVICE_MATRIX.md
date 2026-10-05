# Device matrix

Which tier runs on which hardware, and what accuracy it honestly delivers. Accuracy comes from our own benchmark
(`bench/results/gates.md`, Faro laser ground truth, `benchmark/SET.md`) plus the wider laser-scored set used
for calibration. "Not measured" means we have no data for that cell.

## Tier × hardware

| Tier | Capture hardware | Capture tool | Runs | Tested on |
|---|---|---|---|---|
| Photo | any iPhone 15 or newer (no depth, no poses) | built-in Camera app, 2–8 photos per room, one folder per room | yes | stills from iPad Pro 2020, iPhone 12 Pro Max and iPhone Pro recordings |
| Video | any iPhone 15 or newer | built-in Camera app, one walkthrough clip | yes | the RGB stream of the same recordings |
| LiDAR | iPhone 15 Pro, 16 Pro, 17 Pro (any LiDAR iPhone or iPad) | Stray Scanner (free) | yes | iPad Pro 2020 (ARKitScenes), iPhone 12 Pro Max (MuSHRoom, depth via Polycam), iPhone Pro (own Stray recordings) |
| Processing | Mac with Apple Silicon (16 GB tested) or Linux with Python 3.11+ | `roomscope run <capture>` | yes | M1, 16 GB |

No iPhone 15 or newer is in the benchmark. The phones tested are older, so the phone the examiners bring is
untested. The LiDAR tier carries a 1 % depth-scale prior for an unknown device until
`roomscope calibrate-depth` measures it.

## Honest accuracy per tier (Faro laser, 2026-10-05)

| Quantity | LiDAR | Video | Photo |
|---|---|---|---|
| Ceiling height | 4/4 rooms within 1.5 cm (median 0.8 cm, max 1.4 cm). Spread across repeat captures 0.4–0.6 cm | 8–28 cm off (3–12 %) on 4 captures | 7–28 cm off on 3 captures |
| Wall length, clean laser reference | median 8.7 cm on 21 walls, 9 over 10 cm (ARKitScenes). Errors come from wall placement, not depth (`docs/WALL_ERRORS.md`) | too few cleanly referenced walls to state; wall planes 10–100 cm off | too few cleanly referenced walls to state |
| Brief tolerance | (ceiling 1.5 cm) | walls ±3 %: **not met** | walls ±8 %: **not met** |
| Repeatability (two captures, same room) | 4/12 walls within max(1 cm, 0.5 %) | 0/4 | 0/2 |
| Intervals cover the truth (nominal 90 %) | 13/15 | 4/4, at ±25 % (Fix 3) | 3/3, at ±103 %: ceiling height is barely measurable from eye-level photos |
| Multi-room | splits rooms; the YC apartment gives 6 rooms | **one room only**: COLMAP breaks a multi-room walk into pieces and keeps the largest | all 6 YC rooms reconstructed, but only 1 connected: the photo tier detects no doors, so 5 rooms are placed aside (stitch gate not met) |
| Openings | detected, not measured (no opening ground truth) | as LiDAR | as LiDAR |
| Damage | runs: multi-view vote on the LiDAR surfaces | not run (no per-frame depth) | not run |

## What drives each tier's error
- **LiDAR:**
  - **Depth reads short:** iPad 1.2 %, corrected per device; an unknown device gets a 1 % prior.
  - **The real limit is wall placement:** a wall is sometimes taken from a furniture front or from the
    visible floor edge.
- **Video and photo:** metric scale comes from a monocular/multi-view depth model (Depth Anything 3).
  - **Scale is off by 3–12 %** per capture, and the error is fully correlated across the room.
  - **COLMAP drops frames** in low-texture rooms.
