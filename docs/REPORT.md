# roomscope: technical report

Phone capture in, dimensioned multi-room floor plan out, with an interval on every number. One command per
capture (`roomscope run <capture>`), one JSON contract (`schema/plan.schema.json`) for all three tiers.
Every number below can be regenerated (§8). Where we have no evidence for something, this report says so.

## 1. Architecture
Each tier has its own front end. All three produce the same thing: posed frames, meaning per-frame 3D points
in a camera frame plus a camera-to-world pose. A shared core turns those into the plan.

- **LiDAR** (`frontends/lidar.py`): Stray Scanner export, or ARKitScenes raw. Uses ARKit poses and the
  phone's own depth at confidence 2 only. Depth is divided by a per-device scale (`config/depth_scale.yaml`).
- **Video** (`frontends/video.py`, `recon.py`): COLMAP SfM poses on about 200 frames, then Depth Anything 3
  depth posed on 32 keyframes. Metric scale comes from DA3-Metric.
- **Photo** (`frontends/photo.py`, `core/stitch.py`): each room folder is reconstructed alone (DA3
  multi-view). Rooms are then placed against each other by matching doorways.
- **Shared core** (`core/layout.py`, `drift.py`, `calibrate.py`): runs in this order.
  1. Drift correction (§3).
  2. Fuse the frames and align them to the room's Manhattan axes.
  3. Fit the floor and ceiling planes.
  4. Find wall planes as histogram peaks per facing direction.
  5. Split rooms on a free-space raster, cutting at walls and doorways.
  6. Build the room polygons.
  7. Detect openings from rays that pass through a wall.
  8. Calibrate the intervals with conformal prediction (§5).
- **Damage** (`damage/`, LiDAR tier only):
  - Grounding DINO proposes boxes, SigLIP 2 classifies each crop against damage and clean-surface labels, and
    SAM 2.1 masks the boxes it keeps.
  - The masks are voted across views on a 2 cm grid over the room's walls and ceiling. Depth decides which
    cells each frame can see.
  - Rule tables with public citations (`rules/`) produce concealed-damage flags and scope items.
- **Models:** all run locally and are ungated, with Apache or MIT licences. `scripts/fetch_weights.sh`
  fetches them and `meta.models` discloses them. No API is called.

## 2. Tier design and device matrix
See `docs/DEVICE_MATRIX.md`. In short:
- **LiDAR** is the accurate tier for ceilings: 4/4 rooms within 1.5 cm.
- **Walls** are limited by where the pipeline places them, not by sensor noise (§6).
- **Video and photo** run end to end but miss their accuracy gates. Their metric scale is off by 3–12 %, and
  their intervals widen to say so (§5, Fix 3).
- **Multi-room video is broken:** COLMAP splits the walk into pieces and only the largest is kept.

## 3. Drift handling
Plane-anchored pose-graph correction (`core/drift.py`):
1. The trajectory is cut into 4 s fragments.
2. Each fragment's wall and floor planes are associated with a growing global plane map. Revisiting a wall
   closes the loop.
3. Fragment translations and plane offsets are solved jointly with a soft-L1 loss, plus a 3 cm odometry prior.

`--no-drift` is the ablation.

Measured per frame against the laser on 421337, the defaults do no harm: median camera error 2.4→2.5,
1.9→1.9 and 2.2→1.9 cm. Two earlier settings were rejected: per-fragment yaw and 20 cm association made
poses worse (2.4→8.2 cm). Known limit: a large injected drift is barely removed (16.6→15.5 cm).

**Stitched footprint on the multi-room capture (YC c7d28f72c6, 215 s, 100 m walk), drift on vs off:**

| | rooms | footprint | room areas (m²) | wall/floor plane disagreement |
|---|---|---|---|---|
| drift correction on | 6 | 62.3 m² (58.7–65.9) | 3.7, 5.5, 7.2, 11.9, 13.0, 21.0 | 4.7 → 2.8 cm |
| off (poses as-is) | 5 | 60.1 m² (56.6–63.6) | 3.4, 5.3, 11.6, 12.6, 27.2 | — |

With correction on, the 7.2 m² bathroom is separated from the hallway. With it off, the two merge into one
27.2 m² "room". The footprint moves by 2.2 m² (3.6 %), within both intervals. This capture has no ground truth,
so we cannot say which footprint is closer to the real one, only that correction makes the walls agree better
(plane residual 4.7 → 2.8 cm) and recovers one room.

## 4. Error budget (LiDAR, laser-scored)
| Source | Size | Treatment |
|---|---|---|
| Depth scale (reads short) | −1.1 to −1.3 % of range (iPad), so −3.6 cm on a 3 m ceiling | divided by a scale fitted on other captures; 1 % prior on an unknown device |
| Depth noise on matched surfaces | 0–3 cm | averaged by plane fits |
| ARKit pose error | ~2 cm median | drift correction (do-no-harm) |
| **Wall placement** (wrong surface, or the edge of visible floor) | **10–60 cm on some walls** | not fixed; see `docs/WALL_ERRORS.md` |
| Ceiling with two levels | up to 77 cm when only one level is seen | known failure (MuSHRoom coffee_room) |

Video and photo: metric scale is 3–12 % per capture, fully correlated across the room. It dominates
everything else.

## 5. Calibration
Split conformal on held-out captures (`core/calibrate.py`). The score is |error| / u, with
u = √(σ_raw² + (1 cm)² + (0.5 % · value)²). Quantiles are room-weighted; coverage is reported
leave-one-room-out with Clopper–Pearson intervals.
- **LiDAR:** fitted on ARKitScenes against the earlier nearest-surface reference. On the benchmark set,
  13/15 intervals contain the laser value (nominal 0.9). It still has to be refit against the structural
  reference.
- **Video and photo:** were uncalibrated, with raw ±0.9 cm intervals that never contained the truth. Fix 3
  fits them on 8 laser rooms disjoint from the benchmark (§7). Now 7/7 on the benchmark, at ±25 % (video) and
  ±103 % (photo). The photo width says ceiling height is not measurable from eye-level photos.
- **Run-to-run variation:** the video geometry is not deterministic. A full rerun of one capture moved one wall
  by 23 cm (COLMAP).
- **Damage areas** are labelled uncalibrated: no captured room with real damage exists.

## 6. Ground truth and what the benchmark can show
Walls are scored against a structural laser reference (`eval/laser._structural_wall`): the laser surface that
reaches the ceiling, searched ±1 m across our edge. The earlier reference took the laser surface nearest our
edge, so an edge on a cabinet front was scored against that cabinet front, and one wall's reference length
moved 29 cm between two captures of the same room. With the structural reference the same wall agrees within
0.6 cm. The honest LiDAR wall-length median is 8.7 cm, not the 1.4 cm reported before.

The benchmark set (`benchmark/SET.md`) is real data only:
- a multi-room apartment (YC, no ground truth);
- two rooms each captured twice with Faro laser.

The staged-damage room, tape ground truth for the apartment, opening ground truth and the Polycam
head-to-head are **not done**: they need physical access.

## 7. Fix loop
Full declarations and outcomes are in `fix/DECLARATION.md`; each before/after is tagged and regenerable.
1. **Depth bias:** the ceiling went from −3.2 cm to +0.2 cm average error (prediction met 3/3).
2. **Wall-length intervals:** Mondrian bins; interval score 65→48 cm (4 of 5 predicted numbers met).
3. **Video/photo intervals never contained the truth (the worst gate at 9daf662).**
   - **Root cause:** their intervals were raw plane-fit noise (±0.9 cm). The real error is the reconstruction's
     metric scale, 3–12 %.
   - **Fix:** conformal calibration fitted on 8 laser rooms disjoint from the benchmark.
   - **Coverage:** 0/7 → **7/7** (predicted ≥ 6/7, met).
   - **Width:** video ±25 %, photo ±103 % (predicted 5–15 %, missed: the calibration rooms are worse than the
     benchmark, and photo ceilings are often unseen).
   - **Accuracy:** unchanged, as predicted. Honest intervals, not better numbers. The next fix is metric scale.

## 8. Reproduction and timing
- **Install:** `uv sync --extra video --extra dev`, then `scripts/fetch_weights.sh`, then `roomscope doctor`.
- **Gates:** `python bench/gates.py run && python bench/gates.py score` writes `bench/results/gates.md`.
- **Fix 3:** `python bench/calibrate.py --tier video|photo`.
- **Damage detector check:** `python bench/damage_photos.py`.
- **Model outputs** are cached by input hash, so a rerun replays them; the live path is the same code.

Timing on an M1 with 16 GB, per capture:
- **LiDAR:** ~35 s geometry, plus ~150 s damage on an 80–90 s recording.
- **Video:** ~4.5 min (COLMAP matching ~2 min).
- **Photo:** ~1.5 min per room.

## 9. Known failure modes
- **Mirrors, glass, wet-look surfaces:**
  - *Mitigated:* only confidence-2 LiDAR depth is used, which drops specular and grazing returns.
  - *Mitigated:* damage needs agreement across views, so a reflection that moves between views is rejected.
  - *Not mitigated:* glass walls return no depth, so spaces behind glass merge into one room (vslamlab office
    recording: one 219 m² "room").
- **Low light:**
  - *Mitigated:* damage skips frames that are too dark and says so.
  - *Not mitigated:* geometry has no low-light check.
- **Walls on furniture:** a wall placed on a wardrobe, counter or cabinet front, or on the visible floor edge,
  is 10–60 cm off. `observed_fraction` can still read high there, so the interval does not widen enough.
- **Ceiling with two levels:** if one level is seen only at grazing angles, the wrong level is reported.
- **Multi-room video:** COLMAP fragments the walk; one piece is kept.
- **Out of scope:** an outdoor recording is still turned into "rooms". Nothing checks that a capture is an
  indoor room.
- **Untested device:** no iPhone 15 or newer in the benchmark; the unknown-device depth prior applies.
