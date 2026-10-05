# Benchmark capture manifest

Fill in as you capture. Raw files live under `benchmark/raw/<capture_id>/` (gitignored, sizes too large) and are
packaged for submission by `scripts/package_raw.sh`.

## Required composition (brief §2)
- [ ] Multi-room: ≥3 rooms + a connector (hallway/landing)
- [ ] One furnished room with staged damage across 2 classes (e.g. water stain + drywall hole)
- [ ] Every benchmark room captured at all three tiers (photos as per-room folders)
- [ ] ≥1 room captured twice per tier (repeatability)
- [ ] Laser ground truth for every room (`benchmark/gt/<room>.yaml`)
- [ ] Polycam (free tier) on 2 of the rooms, export + on-screen dimension screenshots, app version noted

## Captures
| capture_id | tier | rooms covered | device | iOS | app + version | repeat # | notes |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## yc/ — own iPhone Pro Stray Scanner recordings of one apartment (no ground truth)
Supplied 2026-10-05 ("Assignment - YC Startup"); copied unchanged to `benchmark/raw/yc/<id>/`
(rgb.mp4 1920×1440 ~46 fps, depth/ + confidence/ 256×192 PNG, odometry.csv, camera_matrix.csv, imu.csv).
Recorded back to back within ~7 min on the same phone, held upright. No laser/tape measurements exist and the
space is not accessible, so these serve walk-in rehearsal, repeatability, drift and cross-tier checks, not accuracy.

| id | name given by the user | length | camera path | closes loop |
|---|---|---|---|---|
| c00a170fe1 | single room | 37 s | 14.5 m | no (3.2 m) |
| 1a8384c3f6 | single scan, floor only | 115 s | 54 m | yes (0.17 m) |
| c7d28f72c6 | single scan, with ceiling | 215 s | 100 m | yes (0.39 m) |

## vslamlab/ — public Stray Scanner recordings by another phone (no ground truth)
Source: https://huggingface.co/datasets/vslamlab/strayscanner (no data licence stated; used only as a format
and robustness check, not redistributed). Unzipped without `__MACOSX/` into `benchmark/raw/vslamlab/<id>/`.
Same export layout as yc/ (odometry.csv with per-frame intrinsics), but fx 1339 px vs 1598 px: a different
iPhone model, so the unknown-device depth prior applies.

| id | content (from the video) | frames | pipeline result 2026-10-05 |
|---|---|---|---|
| 4e41d0a7da | open-plan office/lab, glass-walled rooms, camera pointed mostly at the floor | 3,481 | 1 "room" of 219 m2: glass returns no depth, so nothing separates the spaces; ceiling never seen (reported as not observed) |
| old_parliament_house | **outdoors at night**, around a building | 5,730 | 2 "rooms" (74, 356 m2) with no out-of-scope warning: the pipeline has no check that a capture is an indoor room |

## mushroom/ — MuSHRoom iPhone captures with a Faro laser scan (external accuracy check)
Ren et al., "MuSHRoom: Multi-Sensor Hybrid Room Dataset", arXiv 2311.02778, CC BY 4.0 (cite when used).
Zenodo records 10230733 (`<room>_iphone.tar.gz`) and 10222321 (`<room>_mesh_pd.tar.gz`, Faro Focus 3D X130:
`gt_pd.ply`, `gt_mesh.ply`, `icp_iphone.json`), unpacked into `benchmark/raw/mushroom/room_datasets/<room>/`.
iPhone 12 Pro Max via Polycam: keyframes 738x994 with depth already upsampled and hole-filled by Polycam (no
raw 256x192 LiDAR, no confidence, no IMU), so this checks geometry and intervals on another phone, not raw
LiDAR bias. Converted to Stray layout by `bench/mushroom_to_stray.py` into `benchmark/raw/mushroom_stray/`,
scored by `bench/mushroom_eval.py` (rigid registration to gt_pd.ply; icp_iphone.json not used).

| room | capture | frames | ceiling err | walls covered | wall length MAE | depth scale from laser ceiling |
|---|---|---|---|---|---|---|
| vr_room (5.1 x 4.4 m) | long | 422 | -1.28 cm (covered) | 7/7 | 2.08 cm | 0.9858 (-1.42 %) |
| vr_room | short | 186 | -0.45 cm (covered) | 5/5 | 1.37 cm | 0.9880 (-1.20 %) |
| honka | long | 334 | -0.95 cm (covered) | 8/8 | 6.06 cm (max 21.0) | |
| honka | short | 157 | -1.36 cm (covered) | 8/8 | 3.05 cm | |
| coffee_room | long | 388 | -1.94 cm (covered) | 4/4 | 6.07 cm | |
| coffee_room | short | 134 | **-77.4 cm, NOT covered** | 7/8 | 8.36 cm | |

(2026-10-05, unknown-device prior applied; scale from `roomscope calibrate-depth --dry-run` with the laser
ceiling 3.5157 m. iPad Pro (2020) for comparison: 0.9883.)

coffee_room short is confident garbage: the room has two ceiling levels (a plain section at 2.40 m over
the door end, suspended tiles at 3.22 m elsewhere; laser reference at the centroid 3.22 m). The short capture
barely looked up (steepest frame ~6 deg above level), saw the tiles only at grazing angles (~540 points spread
over 0.2 m), so `pick_level` rejected that level and reported 2.45 m as observed with a +-5 cm interval.
Root cause: `ceiling_height` assumes one flat ceiling per room. Open.
