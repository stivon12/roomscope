<style>
h2, h3 { break-after: avoid; page-break-after: avoid; }
table, tr, li { break-inside: avoid; page-break-inside: avoid; }
h3 { font-size: 10.4pt; margin: 8px 0 3px; }
</style>

# roomscope: Technical Report

roomscope turns a phone capture into a dimensioned multi-room floor plan with a **calibrated interval on every
number**. One command per capture (`roomscope run <capture>`), one JSON contract (`schema/plan.schema.json`) for
all three tiers (photo, video, LiDAR).
- **Ground truth:** every accuracy number is scored against Faro laser scans (`benchmark/SET.md`).
- **Reproduce:** `python bench/gates.py run && python bench/gates.py score` regenerates `bench/results/gates.md`.

### Results at a glance

| Result | Number | Source |
|---|---|---|
| Head-to-head vs Polycam (LiDAR tier, laser-scored) | beat or tie on **9/10 dimensions (90 %; gate 70 %: met)** | `head_to_head.md` |
| Wall positions vs laser on shared dimensions | every one within **1.8 cm** | `head_to_head.md` |
| LiDAR ceiling height, iPad rooms | **2/2 within 0.8 cm** of the laser | `COMPLIANCE.md` G.2 |
| LiDAR ceiling repeatability (same room, two captures) | **2/2 within 1 cm** (0.38 cm, 0.53 cm): met | `gates.md` G.2b |
| LiDAR wall length, clean laser reference | **5/5 within ±3 %**, median **0.6 %** | `gates.md` G.6-lidar |
| Calibrated intervals contain the laser value (nominal 0.9) | video **4/4**, photo **2/2**: met; LiDAR **8/9** (0.89, nominal 0.9) | `gates.md` CAL-* |
| Fix 4 (repeatability) | **all predictions met;** LiDAR repeatability **21 % → 44 %** | `DECLARATION.md`, `gates.md` |
| Drift correction, multi-room apartment | **recovers a room** (6 rooms vs 5 without it) | §3 |

### Gates not met, and why

| Gate | Measured | Cause |
|---|---|---|
| G.2a ceiling ≤ 1.5 cm (LiDAR) | 2/4 rooms | **MuSHRoom iPhone reads ceilings 2.4 % short** (−2.7 / −3.3 cm); Polycam shows the same bias. iPad rooms pass. |
| G.3 repeatability, LiDAR | 4/9 walls (44 %) | **On 421337 the two captures saw different amounts of the entry passage**, so each closes the room in a different place. honka: 4/6. |
| G.3 repeatability, video / photo | 0/4, 0/3 | **Metric scale from the monocular depth model is off 3–12 % per capture.** |
| G.6 walls ±3 % video / ±8 % photo | not measurable (no clean-reference walls) | **Same scale error:** ceilings are 3–12 % off, beyond either tolerance. |
| G.5 photo whole-property stitch | 6/6 rooms reconstructed, **1 connected** | **No doors detected at the photo tier**, so rooms cannot be linked. |
| G.1 openings ≤ 2 cm | not measurable | **No opening ground truth** in any laser scan of the set. |

## 1. Architecture

Each tier has its own front end; every front end produces **posed frames** (per-frame 3D points plus a
camera-to-world pose). One shared core turns posed frames into the plan.

| Stage | Code | What it does |
|---|---|---|
| LiDAR front end | `frontends/lidar.py` | Stray Scanner or ARKitScenes; ARKit poses; confidence-2 depth only, divided by a per-device depth scale |
| Video front end | `frontends/video.py`, `recon.py` | COLMAP poses on ~200 frames; Depth Anything 3 (DA3) depth on 32 keyframes; scale from DA3-Metric |
| Photo front end | `frontends/photo.py`, `core/stitch.py` | Each room folder reconstructed alone (DA3 multi-view, EXIF focal); rooms joined by doors seen from both sides |
| Drift | `core/drift.py` | Plane-anchored pose-graph correction (§3) |
| Layout | `core/layout.py` | Manhattan alignment; floor, ceiling and wall planes; room split on a free-space raster; rectilinear polygons; openings |
| Calibration | `core/calibrate.py` | Split-conformal intervals per tier and quantity (§5) |
| Damage | `damage/`, `rules/`, `scope/` | LiDAR tier only (below) |
| Output | `pipeline.py`, `render.py` | Schema-valid `result.json`, `plan.png` |

**Damage pipeline.** Grounding DINO proposes boxes → SigLIP 2 classifies each crop against damage and
clean-surface descriptions → SAM 2.1 masks the kept boxes → masks are voted across views on a 2 cm grid per
surface, using each frame's depth for visibility → cited rule tables emit flags and scope items (EPA mold tiers,
EPA wicking data, BRE crack categories, 40 CFR 745 lead paint).

**Models.** All local, ungated, Apache-2.0 or MIT; fetched by `scripts/fetch_weights.sh` and listed in each
result's `meta.models`. No API calls.

**Design choice.** A shared core behind thin front ends gives every tier the same contract, plan logic and
calibration. Tiers differ only in the quality of their posed frames, and their intervals carry that difference.

## 2. Tier design and device matrix

| | Photo | Video | LiDAR |
|---|---|---|---|
| Hardware | any iPhone 15+ | any iPhone 15+ | iPhone Pro (LiDAR) |
| Capture | Camera app, 2–8 photos per room, one folder per room | Camera app, one walkthrough | Stray Scanner (free), one recording through all rooms |
| Poses / scale | DA3 multi-view / DA3-Metric | COLMAP / DA3-Metric | ARKit / LiDAR depth |
| Ceiling vs laser | 7–28 cm off (3 rooms) | 8–28 cm off, **3–12 %** (4 rooms) | iPad **within 0.8 cm** (2/2); MuSHRoom iPhone −2.7 / −3.3 cm |
| Ceiling repeat spread | – | – | **0.38 / 0.53 cm** |
| Wall length vs laser | too few clean walls | too few clean walls | benchmark **5/5 within ±3 %**; median **8.7 cm** on 21 walls (wider set) |
| Brief tolerance | walls ±8 %: not met | walls ±3 %: not met | ceiling 1.5 cm: 2/4 |
| Intervals cover truth | **2/2** at ±103 % | **4/4** at ±25 % | **8/9**, ceiling ±3.5 cm |
| Multi-room | 6 rooms, 1 connected (§7) | 1 room (COLMAP splits the walk) | **6 rooms with adjacency** |
| Runtime (M1, 16 GB) | ~13–16 s per room | ~3.5–5 min | ~35 s per room, plus ~2.5 min damage |

Capture steps for every tier are in `CAPTURE_PROTOCOL.md`; the full matrix is `docs/DEVICE_MATRIX.md`.

**Tested devices:** iPad Pro 2020 (ARKitScenes), iPhone 12 Pro Max (MuSHRoom), own iPhone Pro recordings.
**Known limit:** no iPhone 15 or newer is in the benchmark. An unknown device runs with a **1 % depth-scale
prior** until `roomscope calibrate-depth` measures it from one tape distance.

## 3. Drift handling

`core/drift.py`: plane-anchored correction with loop closure on walls.
1. Cut the trajectory into 4 s fragments, where tracking is locally reliable.
2. Match each fragment's wall and floor planes (within 8 cm) to a growing global plane map. Seeing an old wall
   again is the loop closure.
3. Solve all fragment translations and plane offsets jointly: soft-L1 loss against bad matches, 3 cm prior
   towards odometry.
4. Interpolate the corrections between fragments.

On video, COLMAP bundle adjustment already closes loops; the plane step still aligns the walls. On LiDAR
(ARKit poses) it is the only correction.

**Do no harm, measured per frame against the laser:** median camera error 2.4 → 2.5, 1.9 → 1.9 and
2.2 → 1.9 cm on three captures.

**On/off ablation, multi-room apartment (YC, `--no-drift`):**

| Drift correction | Rooms | Stitched footprint (90 % interval) | Wall/floor plane residual |
|---|---|---|---|
| **on** | **6** | 64.0 m² (56.0–72.0) | **2.76 → 2.02 cm** |
| off | 5 | 60.1 m² (56.6–63.6) | – |

**Drift correction recovers a room:** the walls agree better and a **7.18 m² bathroom** becomes its own room;
without it, the bathroom merges into the hallway.
- **Limit:** YC has no ground truth, so footprint accuracy is not scored (next: tape-measure it).
- **Limit:** large drift is only partly removed: on the known-answer test, 16.6 cm of injected drift drops to
  15.5 cm.

## 4. Error budget (LiDAR tier, laser-scored)

| Source | Size | Treatment |
|---|---|---|
| Depth reads short | **−1.1 to −1.3 %** of range (iPad); −3.6 cm on a 3 m ceiling | per-device scale fitted on other captures: iPad −1.17 %, MuSHRoom iPhone −0.32 %; 1 % prior on unknown devices |
| MuSHRoom iPhone ceiling depth | **−2.4 %** at every angle and range (walls within 0.2 %); Polycam shows the same | reported, not corrected: no held-out laser ceiling for this phone |
| Depth noise on surfaces | 0–3 cm | averaged by plane fits |
| ARKit pose error | ~2 cm median | plane-anchored drift correction |
| ARKit heading jump | 15–19° for the first 12 s of 42444946 | fragments > 3° off the room's wall direction dropped, with a warning |
| **Wall placement** (furniture front or visible-floor edge taken as the wall) | **10–60 cm on some walls** | not fixed (`docs/WALL_ERRORS.md`); **dominates the wall error** |
| Two-level ceiling seen from one level | up to 77 cm | known failure (§7) |

**Video and photo:** metric scale is **off 3–12 % per capture**, correlated across the whole room, so it
dominates every other term. Photo ceilings are worse still: eye-level photos rarely see the ceiling.

**Scoring reference.** A wall is scored against the laser surface that reaches the ceiling, searched ±1 m across
our edge, so the reference is independent of where our edge sits. The same physical wall gets the same reference
from two captures **within 0.6 cm**.

## 5. Calibration analysis

**Method:** split conformal prediction on held-out captures.
- **Score:** |error| / u, with u = √(σ_raw² + (1 cm)² + (0.5 % · value)²).
- **Room-weighted quantiles**, so a room with many walls cannot dominate.
- **Coverage check:** leave-one-room-out, with Clopper–Pearson intervals.
- **Mondrian wall bins:** walls whose two neighbours are well observed get their own quantile (Fix 2).
- **Disjoint data:** no calibration room is a benchmark room, at any tier (421337 and honka excluded).

| Tier | Fitted on (disjoint) | Benchmark coverage (nominal 0.9) | Typical half-width | Note |
|---|---|---|---|---|
| LiDAR | 6 ARKitScenes rooms + MuSHRoom vr_room, coffee_room | **8/9** | ceiling ±3.5 cm; walls ±17–48 cm | two-level laser ceilings not scored |
| Video | 10 captures, 8 rooms | **4/4** | ±25 % | held-out coverage 1.00 [0.54–1.00], n = 10 |
| Photo | 8 captures, 7 rooms | **2/2** | ±103 % | held-out coverage not computable (one ceiling per room) |

**Reading.**
- **LiDAR intervals are informative:** ±3.5 cm on a ceiling.
- **Video and photo intervals are honest and wide.** Photo's ±103 % states that eye-level photos cannot measure
  height; before Fix 3 these tiers claimed ±1 cm and missed by 10–30 cm.
- **Walls borrow from ceilings:** with fewer than 5 laser-referenced walls (video 2, photo 0), walls take the
  ceilings' relative quantile, labelled `conformal-transfer` in the output.
- **Damage areas are marked uncalibrated:** no damaged room has been captured to calibrate on.
- **Video geometry is not deterministic:** COLMAP reruns can move a wall by 23 cm; the interval width is stable.

<div style="break-before: page"></div>

## 6. Fix loop story

Every fix follows one protocol: a declaration tagged `fixN-before` (worst gate, root cause with evidence,
predicted numbers) before any fix code, then the fix and a regenerable before/after (`fix/DECLARATION.md`).

| # | Worst gate | Root cause | Fix | Outcome |
|---|---|---|---|---|
| 1 | Ceiling height | **LiDAR depth reads 1.2 % short** | held-out depth scale per device | ceiling mean \|error\| **3.24 → 0.49 cm**, met on **3/3**; wall half of the prediction missed (explained) |
| 2 | Wall-length intervals too wide | **a wall's length error is set by its neighbours' observation** | Mondrian bins | interval score **65.2 → 47.8 cm**; **4 of 5** predictions met |
| 3 | Video/photo intervals never contain the truth (0/7) | **no calibration; real error is metric scale (3–12 %), intervals were plane-fit noise (±0.9 cm)** | conformal calibration on 8 disjoint laser rooms | coverage **0/7 → 7/7** (predicted ≥ 6/7: met); accuracy unchanged, as predicted; width ±25 % / ±103 % vs 5–15 % predicted (calibration rooms are harder) |
| 4 | Repeatability (LiDAR 3/14) | **floor-boundary notches cut the same wall into different pieces per capture** | merge floor-boundary edges up to furniture depth (0.8 m) | **all predictions met** (below) |

**Fix 4 in detail**, the latest loop:
- **Evidence:** depth and poses repeat (ceilings agree to 0.5–0.6 cm; well-observed walls within ±2 cm of the
  laser in both captures), but the outlines do not. On honka a 0.42 m notch split the long wall: **5.39 m vs
  5.83 m**. After the fix it reads **5.86 m vs 5.83 m**.
- **Results (current gates):** repeatability **21 % → 44 %** (predicted ≥ 40 %); honka pair **4/6**
  (predicted ≥ 4); floor-boundary edges **7 → 3** (predicted ≤ 3); head-to-head vs Polycam **90 %** (predicted
  ≥ 70 %); **no** laser-scored wall more than 2 cm worse.
- **Not fixed, as declared:** 421337's remaining edges border floor the camera never saw, deeper than furniture.
  **Next fix:** close an unobserved side from structure (ceiling edge, wall-line continuation).

Reproduce every round on the current code (each fix switched off, then on): `python fix/reproduce.py 1|2|3|4`.

## 7. Known failure modes

- **Photo stitch connects 1 of 6 rooms.** All 6 rooms are reconstructed; 5 are "placed aside" (adjacency
  unknown) and the output says so.
  - **Cause: no doors detected at the photo tier.** Door detection needs depth seen through the doorway, which
    a handful of photos per room rarely provides; this capture lacks the protocol's doorway shots.
  - **Next:** recapture with the doorway shots.
- **Multi-room video:** COLMAP fragments the walk on plain walls and doorways; the largest piece (one room) is kept.
- **Unseen sides of a room:** where the floor was never seen up to a wall deeper than furniture, the edge stays at
  the floor boundary, marked inferred (`observed_fraction` 0, widened interval). Furniture fronts with a wall
  within 0.8 m snap to the wall.
- **Two-level ceilings:** if one level is seen only at grazing angles, the other level is reported.
- **Mirrors, glass, wet-look surfaces:**
  - **Mitigated:** only confidence-2 LiDAR depth is kept, dropping specular and grazing returns; damage requires
    agreement across views, so a moving reflection is rejected.
  - **Not mitigated:** glass returns no depth, so spaces behind glass merge (a glass-walled office became one
    **219 m²** "room").
- **Low light:** damage skips frames that are too dark and says so; geometry has no low-light check (stated in
  the protocol).
- **Out of scope:** `meta.scope_check` flags captures that are not an enclosed indoor space (no observed ceiling
  and < 20 % of walls observed). It flags an outdoor walk and the glass-walled office and passes every benchmark
  run; flagged captures still produce output, with warnings.
- **Benchmark gaps** (next step: a physical capture visit): no iPhone 15 or newer; no staged-damage room; no ground
  truth for the multi-room capture; no opening ground truth (openings detected but unmeasured).
