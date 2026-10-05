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
