# roomscope: Technical Report

roomscope turns a phone capture into a dimensioned multi-room floor plan, with a calibrated interval on every
number. One command per capture (`roomscope run <capture>`); one JSON contract (`schema/plan.schema.json`) for
all three tiers.
- **Measurements:** every accuracy number is scored against Faro laser scans on a real-data benchmark set
  (`benchmark/SET.md`).
- **Regeneration:** `python bench/gates.py run && python bench/gates.py score` reproduces
  `bench/results/gates.md`.
- **Known limits** are stated where they apply, with the next step.

## 1. Architecture

Each tier has its own front end, and every front end produces the same thing: posed frames (per-frame 3D
points plus a camera-to-world pose). One shared core turns posed frames into the plan.

| Stage | Code | What it does |
|---|---|---|
| LiDAR front end | `frontends/lidar.py` | Stray Scanner export (or ARKitScenes raw). ARKit poses, the phone's own depth at confidence 2 only, divided by a per-device depth scale (`config/depth_scale.yaml`). |
| Video front end | `frontends/video.py`, `recon.py` | COLMAP SfM poses on ~200 frames; Depth Anything 3 depth on 32 keyframes, posed by COLMAP; metric scale from DA3-Metric. |
| Photo front end | `frontends/photo.py`, `core/stitch.py` | Each room folder is reconstructed alone (DA3 multi-view, EXIF focal length). Rooms are placed into one plan by matching doors seen from both sides. |
| Drift | `core/drift.py` | Plane-anchored pose-graph correction (§3). |
| Layout | `core/layout.py` | Fuse and align to the room's Manhattan axes; floor and ceiling planes; wall planes; rooms split on a free-space raster; rectilinear room polygons; openings from rays seen through a wall. |
| Calibration | `core/calibrate.py` | Split-conformal intervals per tier and quantity (§5). |
| Damage | `damage/`, `rules/`, `scope/` | LiDAR tier only (details below). |
| Output | `pipeline.py`, `render.py` | `result.json` (schema-valid), `plan.png`. |

**Damage pipeline:**
1. Grounding DINO proposes boxes.
2. SigLIP 2 classifies each crop against damage and clean-surface descriptions.
3. SAM 2.1 masks the boxes it keeps.
4. The masks are voted across views on a 2 cm grid over each wall and ceiling. The frame's own depth decides
   which cells it can see.
5. Rule tables with public citations produce the flags and scope items: EPA mold tiers, EPA wicking data, BRE
   crack categories, and 40 CFR 745 for lead paint.

**Models and constraints.** Every model runs locally and is ungated, under an Apache-2.0 or MIT licence. They
are fetched by `scripts/fetch_weights.sh` and listed in each result's `meta.models`. No API is called.

**Design choice.** A shared core behind thin front ends means every tier gets the same contract, the same
plan logic and the same calibration machinery. A tier differs only in how good its posed frames are. Its
intervals then carry that difference.

## 2. Tier design and device matrix

| | Photo | Video | LiDAR |
|---|---|---|---|
| Hardware | any iPhone 15+ | any iPhone 15+ | iPhone Pro (LiDAR) |
| Capture (`CAPTURE_PROTOCOL.md`) | Camera app, 2–8 photos per room, one folder per room | Camera app, one walkthrough | Stray Scanner (free), one recording through all rooms |
| Poses / scale from | DA3 multi-view / DA3-Metric | COLMAP / DA3-Metric | ARKit / LiDAR depth |
| Ceiling vs laser | 7–28 cm off on 3 rooms | 8–28 cm off (3–12 %) on 4 | iPad: 2/2 within 0.8 cm; MuSHRoom iPhone: −2.7 / −3.3 cm (ceilings read 2.4 % short, shared by Polycam); repeat spread 0.5–0.6 cm |
| Wall length vs laser | too few clean walls to state | too few clean walls to state | median 8.7 cm on 21 walls (wider ARKitScenes set) |
| Brief tolerance | ±8 % walls: **not met** | ±3 % walls: **not met** | (ceiling 1.5 cm: 2/4) |
| Intervals cover the truth | 4/4, at ±103 % | 4/4, at ±25 % | 8/9 (ceiling ±3.5 cm) |
| Multi-room | 6 rooms reconstructed; 1 connected (§7) | 1 room: COLMAP splits the walk | 6 rooms, with adjacency |
| Runtime (M1, 16 GB) | ~15 s per room (~90 s cold) | ~4–5 min | ~35 s per room cold, plus ~2.5 min damage |

**Tested phones.** iPad Pro 2020 (ARKitScenes), iPhone 12 Pro Max (MuSHRoom) and our own iPhone Pro
recordings. Known limit: no iPhone 15 or newer is in the benchmark, so an unknown device runs with a 1 %
depth-scale prior until `roomscope calibrate-depth` measures it from one tape distance. Full matrix:
`docs/DEVICE_MATRIX.md`.

## 3. Drift handling

`core/drift.py` uses plane-anchored correction with loop closure on walls:
1. The trajectory is cut into 4 s fragments, where tracking is locally reliable.
2. Each fragment's wall and floor planes are matched, within 8 cm, against a growing global plane map. Seeing
   an old wall again (re-entering a room, ending at the start) is the loop closure.
3. All fragment translations and plane offsets are solved jointly. A soft-L1 loss stops one bad match from
   dragging the solution; a 3 cm prior pulls towards the odometry.
4. The corrections are interpolated between fragments.

**What COLMAP adds.** On video, COLMAP's matching and bundle adjustment already close loops inside one
connected reconstruction. The plane step still aligns the walls themselves. It is the only correction on
LiDAR, where poses come from ARKit.

**Measured per frame against the laser.** The defaults do no harm on good poses: median camera error 2.4→2.5,
1.9→1.9 and 2.2→1.9 cm on three captures. Per-fragment yaw and 20 cm matching were rejected because they
degraded good poses (2.4→8.2 cm). Known limit: on the known-answer test, 16.6 cm of injected drift is reduced
to 15.5 cm.

**Ablation on the multi-room capture** (YC, 215 s, 100 m walk, `--no-drift`):

| | rooms | stitched footprint | wall/floor plane disagreement |
|---|---|---|---|
| drift correction on | 6 | 62.3 m² (58.7–65.9) | 4.7 → 2.8 cm |
| off (poses as-is) | 5 | 60.1 m² (56.6–63.6) | — |

Drift correction recovers a room: with it on, the walls agree better and a 7.2 m² bathroom is reconstructed as
its own room; with it off, that bathroom merges into the hallway. Known limit: this capture has no ground truth,
so footprint accuracy is not scored. Next: tape-measure the apartment (`benchmark/SET.md`).

## 4. Error budget (LiDAR tier, laser-scored)

| Source | Size | Treatment |
|---|---|---|
| Depth reads short | −1.1 to −1.3 % of range (iPad); −3.6 cm on a 3 m ceiling | divided by a per-device scale measured on other captures (iPad 0.9883; MuSHRoom iPhone 0.9968 from held-out vr_room); 1 % prior on unknown devices |
| Ceiling depth on the MuSHRoom iPhone | −2.4 % at every angle and range (walls within 0.2 %); Polycam shows the same | reported, not corrected: no held-out laser ceiling for this phone |
| Depth noise on matched surfaces | 0–3 cm | averaged by plane fits |
| ARKit pose error | ~2 cm median | plane-anchored drift correction (do no harm) |
| ARKit heading jump | 15–19° for the first 12 s of 42444946 | 4 s fragments more than 3° off the room's wall direction are dropped, with a warning |
| **Wall placement**: wall taken from a furniture front or the visible-floor edge | **10–60 cm on some walls** | not fixed (`docs/WALL_ERRORS.md`); dominates the wall error |
| Two-level ceiling seen from one level | up to 77 cm | known failure |

**Video and photo.** The reconstruction's metric scale is off by 3–12 % per capture, and the error is shared
across the whole room, so it dominates every other term. Photo ceilings are often worse: eye-level photos
rarely show the ceiling.

**Independent wall reference.** The scorer takes as reference the laser surface that reaches the ceiling,
searched ±1 m across our edge, so it is independent of where our edge sits. It replaced a reference that took
the laser surface nearest our edge, which scored an edge on a cabinet front against that cabinet front.
- **Consistency check:** the same physical wall gets the same reference from two captures, within 0.6 cm
  (the old reference moved it by 29 cm).
- **Result:** the LiDAR wall-length median is **8.7 cm**, correcting the optimistic 1.4 cm reported before.

## 5. Calibration analysis

**Method.** Split conformal prediction on held-out captures.
- **Score:** |error| / u, with u = √(σ_raw² + (1 cm)² + (0.5 % · value)²).
- **Quantiles:** room-weighted, so a room with many walls cannot dominate.
- **Coverage check:** leave-one-room-out, with Clopper–Pearson intervals.
- **Wall bins:** walls whose two neighbouring walls are well observed get their own quantile (Mondrian bins,
  fix 2).
- **Disjoint data:** video and photo calibration rooms never include a benchmark room. LiDAR is now refit the same way (rooms 421337 and honka excluded).

| Tier | Fitted on | Benchmark coverage (nominal 0.9) | Typical half-width | Note |
|---|---|---|---|---|
| LiDAR | 6 ARKitScenes rooms + MuSHRoom vr_room, coffee_room (disjoint) | 8/9 | ceiling ±3.5 cm; walls ±17–48 cm | structural wall reference; two-level laser ceilings are not scored (as stepped walls) |
| Video | 10 captures, 8 rooms (disjoint) | 4/4 | ±25 % | held-out coverage 1.00 [0.54–1.00] (n=10) |
| Photo | 8 captures, 7 rooms (disjoint) | 4/4 | ±103 % | held-out coverage not computable (one ceiling per room) |

**Reading.**
- **LiDAR** intervals are informative.
- **Video and photo** intervals are honest and wide: photo's ±103 % states that these photos cannot measure
  height. Before fix 3 they claimed ±1 cm and missed by 10–30 cm.
- **Walls borrow from ceilings.** With fewer than 5 laser-referenced walls (video 2, photo 0), walls take the
  ceilings' *relative* quantile. This is labelled `conformal-transfer` in the output.
- **Damage areas** are marked uncalibrated: there is no captured damaged room to calibrate on.
- **Run-to-run variation:** video geometry is not deterministic. One full rerun moved a wall by 23 cm
  (COLMAP).

## 6. Fix loop story

The protocol for every fix: a declaration committed and tagged before any fix code, a predicted number, then
the shipped fix and a regenerable before/after (`fix/DECLARATION.md`, tags `fixN-before` / `fixN-after`).
1. **Ceiling depth bias.** LiDAR depth reads 1.2 % short.
   - **Fix:** a held-out depth scale per device.
   - **Result:** ceiling error −3.2 → +0.2 cm average, prediction met on 3/3. The wall half of the prediction
     was wrong, and the declaration explains why.
2. **Wall-length intervals were too wide to be useful.**
   - **Root cause:** a wall's length error is set by its neighbouring walls' observation, not its own.
   - **Fix:** Mondrian bins.
   - **Result:** interval score 65 → 48 cm; 4 of 5 predicted numbers met.
3. **Video and photo intervals never contained the truth** (0/7, the worst gate on the benchmark at 9daf662).
   - **Root cause:** those tiers had no calibration. Their intervals were plane-fit noise (±0.9 cm), while
     their real error is metric scale (3–12 %).
   - **Fix:** conformal calibration on 8 laser rooms disjoint from the benchmark.
   - **Results:** coverage **7/7** (predicted ≥ 6/7: met); interval width ±25 % video, ±103 % photo (predicted
     5–15 %: missed, because the calibration rooms are worse than the benchmark and photo ceilings are often
     unseen); accuracy unchanged (as predicted).
   - **After-run:** calibration is the fix's only changed input, so it was re-applied to the finished runs;
     one full rerun confirmed the same interval width.
   - **Caught before the after-run:** transferring the normalised (not relative) quantile from ceilings to
     walls produced a −1.5 to 7.3 m wall; it was replaced.
4. **The same room measured twice gave different walls** (LiDAR repeatability 3/14, the worst gate).
   - **Root cause (per-wall audit):** depth repeats (ceilings agree to 0.6 cm, observed walls within ±2 cm of the
     laser in both captures), but floor-boundary notches cut the same wall into different pieces per capture.
   - **Fix:** floor-boundary edges up to furniture depth (0.8 m) are merged, not reported as walls.
   - **Results:** repeatability **4/10 (40 %)**, honka pair 4/6 (predicted ≥ 40 %: met); floor-boundary edges
     7 → 3; head-to-head **7/8 (88 %)** (predicted ≥ 70 %: met); no scored wall worse. 421337 stays 0/4: its
     remaining edges border floor never seen, the declared next fix.

**Also fixed while building the benchmark:** the photo tier dropped rooms whose seen floor ended in a diagonal.
The diagonal is a visibility limit, so it is now replaced by axis-aligned legs that snap to walls. All photo runs
produce rooms; LiDAR walls are unchanged.

## 7. Known failure modes

- **Photo stitch connects 1 of 6 rooms.** On the multi-room capture all 6 rooms are reconstructed; 5 are
  "placed aside" (not stitched, adjacency unknown), and the output says so.
  - **Cause:** the photo tier has never detected a door. Door detection needs depth seen through the doorway,
    which 7–8 photos per room rarely provide, and our photos for that capture lack the doorway shots the
    protocol asks for.
  - **Status:** photo whole-property stitch gate **not met**. Next: capture with the protocol's doorway shots.
- **Multi-room video:** COLMAP fragments the walk on plain walls and doorways; the largest piece (one room) is kept.
- **Unseen sides of a room:** where the camera never saw the floor up to a wall deeper than furniture, the edge
  stays at the floor boundary, marked inferred (`observed_fraction` 0, widened interval). Furniture fronts with a
  structural wall within 0.8 m snap to the wall (Fix 4 and the structural snapping).
- **Two-level ceilings:** if one level is seen only at grazing angles, the other level is reported.
- **Mirrors, glass, wet-look surfaces:**
  - **Mitigated:** only confidence-2 LiDAR depth is kept, which drops specular and grazing returns.
  - **Mitigated:** damage needs agreement across views, so a reflection that moves between views is rejected.
  - **Not mitigated:** glass walls return no depth, so spaces behind glass merge (a glass-walled office became
    one 219 m² "room").
- **Low light:** damage skips frames that are too dark and says so. Geometry has no low-light check; the protocol
  states this.
- **Out of scope:** a scope check (`meta.scope_check`) flags captures that are not an enclosed indoor space: no
  observed ceiling and under 20 % of walls observed. It flags an outdoor walk and a glass-walled office, and passes
  all 29 benchmark runs. Flagged captures still produce output, with warnings.
- **Untested hardware:** no iPhone 15 or newer is in the benchmark.
- **Head-to-head vs Polycam** (`bench/results/head_to_head.md`): beat or tie on **9 of 10 dimensions (90 %, gate
  70 %: met)**, after structural-wall snapping, the measured per-device depth scale, Fix 4 and the heading-jump
  filter. Every shared wall position is within 1.8 cm of the laser; the one loss is coffee_room W1 (+1.8 vs +1.2 cm).
- **Benchmark gaps** (next step: a physical capture visit): no staged-damage room; no ground truth for the
  multi-room capture; no opening ground truth (the opening gate is unmeasured).
