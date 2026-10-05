<style>
h3 { font-size: 10.2pt; margin: 9px 0 2px; color: #222; }
.pb { page-break-before: always; }
td:last-child { white-space: nowrap; }
</style>

# Fix loop: declarations and outcomes

Each round names the worst gate in our own benchmark, states a root cause with evidence, declares the fix
and a predicted number **before** the fix is coded, then ships it with a regenerable before/after.
Every number is scored against the Faro laser scans. Misses are reported as misses.

| fix | gate (worst at the time) | before → after | predictions |
|---|---|---|---|
| **1** LiDAR depth scale | ceiling height ≤ 1.5 cm (LiDAR) | mean \|ceiling error\| **3.24 → 0.49 cm**, all 3 captures within 1 cm | ✓ ceiling met · ✗ walls missed |
| **2** wall-length calibration bins | calibrated intervals (LiDAR wall length) | interval score **65.2 → 47.8 cm**; supported half-width 19.7 → 15.7 cm | ✓ 4 of 5 met · ✗ 1 missed by 0.7 cm |
| **3** video/photo calibration | interval coverage (video, photo) | intervals containing the laser value **0/7 → 7/7** | ✓ coverage met · ✗ widths wider than predicted |
| **4** floor-boundary notches | G.3 repeatability (LiDAR) | walls within tolerance **21 % → 40 %**; head-to-head vs Polycam **69 % → 88 %** | ✓ **all 5 met** |

## Fix 1: LiDAR depth reads short (ceiling-height gate)

*Retrospective record:* the declaration was written before the fix was coded but not committed ahead of it,
so git cannot prove the ordering. Fixes 2–4 follow the strict protocol (declaration committed and tagged
`fixN-before` before any fix code).

### 1. Worst gate
Ceiling height, gate ≤ 1.5 cm. On 42444946 the ceiling read **−3.0 cm** vs the laser (3.03 vs 3.06 m),
and walls sat 2–4 cm inside the laser walls.

### 2. Root cause and evidence
The iPad LiDAR reads distances about **1.2 % short**. Measured per pixel without poses, `lowres_depth` vs
the laser-rendered `highres_depth` of the same frame (`src/roomscope/eval/depth_bias.py`):
- median relative error **−1.30 %, −1.10 %, −1.23 %** on 42444946/49/50, flat across 0.5–4 m;
- consistent with Zea & Hanebeck (JAIF 2022/23: "always negative, 1–2 % of range").

Floor and ceiling are both pulled toward a camera held between them, so a 3 m ceiling shrinks by ~3.6 cm.

### 3. Fix
Divide depth by a scale fitted on *other* captures (default: depth 1.17 % short, fitted on
42444949 + 42444950; `config/depth_scale.yaml`), applied at load time before drift correction.
`--no-depth-correction` restores the old behaviour for the ablation.

### 4. Predicted
- |ceiling error| **≤ 1 cm** on the scored capture;
- mean wall offset **≤ 2 cm** with drift off.

### 5. Outcome
Each capture is scored with a scale fitted on the other two only (held out). Regenerate:
`python fix/reproduce.py 1` → `fix/results/fix1.md`.

| capture | held-out scale (depth short by) | ceiling error cm | wall-plane MAE cm | wall-length MAE cm |
|---|---|---|---|---|
| 42444946 | 1.17 % | −3.23 → **+0.15** | 0.91 → 1.62 | 1.28 → 1.80 |
| 42444949 | 1.27 % | −2.66 → **+0.55** | 5.40 → 3.49 | 9.63 → 9.01 |
| 42444950 | 1.20 % | −3.82 → **−0.76** | 3.69 → 3.00 | 7.36 → 6.41 |

| prediction | measured | |
|---|---|---|
| \|ceiling error\| ≤ 1 cm | **all 3 captures**, mean \|error\| 3.24 → **0.49 cm** | ✓ **met** |
| mean wall offset ≤ 2 cm | 42444946 only (and there 0.91 → 1.62 cm, a seventh wall found); 42444949/50 improve but stay > 2 cm | ✗ **missed** |

**Why the wall half missed: it assumed the 2–4 cm inward wall offset of an earlier pipeline version, which
later wall-selection fixes had already removed.** The walls still above 2 cm on 42444949/50 are the
barely-observed entry-alcove walls (16 and 11 cm off); that error is not depth scale.

<div class="pb"></div>

## Fix 2: wall-length intervals are honest but uselessly wide (calibration gate)

Reproduce on the current code: `python fix/reproduce.py 2` → `fix/results/fix2.md` (bins off vs on, same records).

**On today's pipeline the bins no longer pay off:** with the refit records (20 walls, only 4 "inferred"), one quantile
scores better (interval score 112 cm) than the supported bin (140 cm). Later wall-placement fixes removed most of
the poorly supported walls the bins were built for.

### 1. Worst gate
Calibrated intervals on every number, overconfidence penalised. LiDAR wall-length intervals cover
(leave-one-room-out 0.92 [0.81–0.98]) but are useless: median half-width **19.7 cm** for a median error of
4.8 cm (width/error 4.1), mean interval score **65.2 cm** (51 walls, 7 rooms).

### 2. Root cause and evidence
A wall's length is the distance between its two neighbouring walls, so its error is set by **the
neighbours**, not by the wall itself (`bench/wall_audit.py`):
- every wall with |error| ≥ 9 cm has exactly one neighbour 10–20 cm off while the other is within ~1 cm;
  8 of those 11 bad neighbours were observed over ≤ 20 % of their length (placed from the floor extent,
  which furniture cuts short);
- over all 51 walls, error vs the less-observed neighbour's observed fraction: Spearman −0.53 (p = 5e-5);
- both neighbours ≥ 30 % observed: median error **1.4 cm** (n = 31); otherwise **13.7 cm** (n = 20).

One global conformal quantile is set by the second group and applied to the first. The wall's *own*
observed fraction does not predict error (Spearman 0.01).

*Alternative rejected:* placing inferred walls from the ceiling edge. Over the 11 bad neighbour walls the
ceiling edge is within 5 cm of the laser wall on only 4, missing on 5, and 18–29 cm off on 2. (These
numbers were corrected later; the rejection stands.)

### 3. Fix
Mondrian (class-conditional) conformal on wall length with two bins: **corner-supported** (both
neighbours ≥ 30 % observed) and **inferred** (otherwise), each with its own room-pooled quantile. The bin
is computed from the result at run time; inferred walls are labelled in the output.

### 4. Predicted (leave-one-room-out, same 51 walls)
- supported bin: median half-width **10–15 cm**, coverage ≥ 0.85;
- inferred bin: median half-width 20–30 cm, coverage ≥ 0.85;
- mean interval score **≤ 50 cm**.

Declared risk: ~19 % of supported walls are still above 8 cm, so that bin's 90 % quantile cannot be small.
The fix makes intervals adaptive; it does not make the geometry more accurate.

### 5. Outcome

| | predicted | measured (51 walls / 7 rooms) | |
|---|---|---|---|
| supported bin, median half-width | 10–15 cm | **15.7 cm** (from 19.7) | ✗ **missed by 0.7 cm** |
| supported bin, coverage | ≥ 0.85 | 0.94 [0.79–0.99] | ✓ met |
| inferred bin, median half-width | 20–30 cm | 20.9 cm | ✓ met |
| inferred bin, coverage | ≥ 0.85 | 0.95 [0.75–1.00] | ✓ met |
| mean interval score | ≤ 50 cm | **47.8 cm** (from 65.2) | ✓ **met** |

**Why the supported bin stayed wide: 6 of its 31 walls are still > 8 cm off, and their small raw
uncertainties inflate the normalised scores, so its quantile rose (7.05 → 8.63).** The declared risk
anticipated this tail but underestimated it. Next lever: what those 6 walls share (a neighbour fitted to
a front surface while the dominant laser surface lies behind it).

<div class="pb"></div>

## Fix 3: video and photo intervals never contain the truth (calibration gate)

Reproduce on the current code: `python fix/reproduce.py 3` → `fix/results/fix3.md` (raw vs conformal intervals, same records).

### 1. Worst gate
Calibrated intervals contain the laser value for **0 of 4 video and 0 of 3 photo** measurements
(nominal 0.9; LiDAR 13/15). Video/photo accuracy and repeatability gates also fail, but every one of those
numbers is reported with a near-zero interval. A wrong number with a tight interval is the "confident
garbage" the brief penalises most.

### 2. Root cause and evidence
The video and photo intervals are raw plane-fit statistics (~±0.9 cm on a ceiling, ~±8 cm on a wall):
`config/calibration.json` had an entry for LiDAR only, so these tiers were never calibrated. Their real
error is the **metric scale of the RGB reconstruction**, a fully correlated relative error a plane fit cannot
see. On the benchmark set:
- ceilings −7.7, −12.6, −27.4, −28.1 cm (video) and −27.7, −10.0, +7.4 cm (photo), i.e. **3–12 %**;
- wall offsets 10–100 cm.

### 3. Fix
Split-conformal calibration for video and photo on laser-scored captures **disjoint from the benchmark
set**: ARKitScenes 42897521, 42897647, 42897501, 42897545, 42898811, 42898818 and MuSHRoom vr_room and
coffee_room (long and short); 10 captures, 8 rooms. Benchmark rooms (421337, including 42444950, and
honka) are excluded. Normalised score u = sqrt(σ_raw² + (1 cm)² + (0.5 % · value)²), so q scales the
dominant relative term. No pipeline or geometry change.

### 4. Predicted
- benchmark set, video + photo intervals containing the laser value: **≥ 6 of 7** (from 0 of 7);
- leave-one-room-out coverage on the calibration rooms: ≥ 0.80 per tier;
- median ceiling half-width: **5–15 % of the value** (from ~0.3 %);
- accuracy (±3 % / ±8 %) and repeatability: unchanged. This fix makes intervals honest, not measurements
  more accurate.

### 5. Outcome

| | predicted | measured | |
|---|---|---|---|
| video + photo intervals containing the laser value | ≥ 6 of 7 | **7 of 7** (from 0 of 7) | ✓ **met** |
| LORO coverage, calibration rooms | ≥ 0.80 per tier | video ceilings 1.00 [0.54–1.00] (n = 10); photo not computable (one ceiling per room) | ✓ met (video) |
| median ceiling half-width | 5–15 % | video **±25 %**, photo **±103 %** | ✗ **missed, wider** |
| accuracy gates and repeatability | unchanged | unchanged (repeatability video 0/4, photo 0/2; walls not measurable) | unchanged |

**Why the widths missed: the prediction came from the benchmark's own errors (3–12 %), but the calibration
rooms are harder** (median ceiling error 20 cm for video, 52 cm for photo; eye-level photos rarely see the
ceiling). The wide photo intervals are the correct answer: they say these photos cannot measure ceiling
height, where before they claimed ±1 cm and were 10–30 cm wrong.

Shipping details, both labelled in the output: with fewer than 5 laser-referenced walls (video 2, photo 0),
walls borrow the ceiling's relative quantile (|error| ÷ value); lengths are floored at 0. The after-numbers
re-apply the new calibration to the finished before-runs (geometry code unchanged); a full rerun of video
42444946 gives the same ceiling interval (±0.69 m), also containing the laser value; its geometry differs by 1 cm on
the ceiling and 23 cm on one wall because COLMAP is not deterministic between runs.

Next lever: metric scale from the RGB depth model, the root cause of those widths.

<div class="pb"></div>

## Fix 4: the same room measured twice gives different walls (repeatability gate)

Reproduce on the current code: `python fix/reproduce.py 4` → `fix/results/fix4.md` (notch merge off vs on).

### 1. Worst gate
G.3 repeatability: wall lengths from two captures of one room within max(1 cm, 0.5 %). LiDAR scores
**3/14 (21 %)**, the lowest gate on the benchmark set:

| tier | pair | walls within tolerance | median length difference | ceiling difference |
|---|---|---|---|---|
| LiDAR | 42444946 / 42444949 | **0/8** | 14.0 cm | 0.6 cm |
| LiDAR | honka_long / honka_short | 3/6 | 2.1 cm | 0.5 cm |
| video | both pairs | 0/4 | 12–96 cm | 1.3–4.6 cm |
| photo | both pairs | 0/3 | 35–105 cm | 17–27 cm |

Scope: LiDAR. Video and photo disagree because per-capture metric scale is off by 3–12 %, which no outline
change can bring to 0.5 %.

### 2. Root cause and evidence (per-wall audit of both pairs vs the laser)
- **Depth and poses repeat.** Ceilings agree to 0.5–0.6 cm within each pair; every well-observed wall is
  within ±2 cm of the laser in *both* captures.
- **The outlines do not.** 42444946 has 10 edges, 42444949 has 12; of these, 2 and 4 are floor-boundary
  edges with no wall plane behind them (the visible floor stopped at furniture, not at a wall).
- **Each notch cuts a real wall into different pieces per capture.** In honka, one 0.42 m floor-boundary
  notch in honka_long splits the long wall: **5.39 m** there vs **5.83 m** in honka_short. The same notch
  causes 3 of the 4 head-to-head losses.

### 3. Fix
A floor-boundary edge (no wall plane, not shared with another room) is not a wall. If the notch is no
deeper than the furniture reach already used to find walls (`STRUCT_REACH`, 0.8 m), the edge is removed and
its two neighbours merge, the better-supported plane keeping the position. Deeper ones stay, marked
inferred with a widened interval. No other threshold changes.

### 4. Predicted (LiDAR, benchmark set)
- G.3-lidar **≥ 40 %** (from 21 %); honka pair ≥ 4 walls within tolerance;
- floor-boundary edges on the four laser captures: 7 → **≤ 3**;
- head-to-head vs Polycam **≥ 70 %** (from 69 %);
- do no harm: no laser-scored wall position > 2 cm worse; ceiling and calibration gates unchanged.

### 5. Outcome

| | before | predicted | after | |
|---|---|---|---|---|
| G.3-lidar, walls within tolerance | 3/14 (21 %) | ≥ 40 % | **4/10 (40 %)** | ✓ **met** |
| honka pair | 3/6, median 2.1 cm | ≥ 4 | **4/6, median 1.1 cm** | ✓ **met** |
| floor-boundary edges, 4 laser captures | 7 | ≤ 3 | **3** | ✓ **met** |
| head-to-head vs Polycam | 9/13 (69 %) | ≥ 70 % | **7/8 (88 %)**, gate reached | ✓ **met** |
| laser-scored wall positions > 2 cm worse | – | 0 | **0** | ✓ **met** |
| ceiling gate / interval coverage | 2/4 / 10/12 (83 %) | unchanged | 2/4 / 8/9 (89 %) | ✓ ceiling unchanged; coverage 83 → 89 % |

- **honka:** both captures now give the same six walls; the long wall reads **5.86 m and 5.83 m** (was 5.39
  and 5.83 m).
- **Head-to-head:** 8 shared dimensions, down from 13. Without the notch, three honka walls run along a
  two-level laser surface, so they are reported as steps (lower-bound errors) and not shared. The one
  remaining loss is a coffee_room length (+1.4 cm vs Polycam +0.8 cm).

**Not fixed: the 421337 pair stays 0/4 because its remaining floor-boundary edges (1.6, 0.87, 1.5 m) are
deeper than furniture and border floor the camera never saw, so each capture closes the room in a
different place.** The rule keeps them, as declared. Next lever: close an unobserved side from structure
(the ceiling edge, or the wall line continuing) instead of the floor boundary.
