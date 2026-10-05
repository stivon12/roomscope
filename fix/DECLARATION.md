# Fix loop

## Fix 1: LiDAR depth reads short (ceiling-height gate)

**Disclosure.** The declaration below was written in the planning document before the fix was coded
(diagnostics in f29cf70 / c80bb5d, fix in a72bc9f), but it was not committed to git ahead of the fix, so
the repository cannot prove the ordering. It is reported as a retrospective record. Fix 2 follows the
strict protocol: declaration committed and tagged `fix2-before` before any fix code.

**Worst gate at the time.** Ceiling height, gate <= 1.5 cm. On 42444946 vs the Faro laser the ceiling
was -3.0 cm (3.029 vs 3.059 m), and walls sat 2-4 cm inside the laser walls.

**Root cause (evidence).** The iPad LiDAR reads distances about 1.2 % short. Measured without poses:
per pixel, `lowres_depth` vs the laser-rendered `highres_depth` of the same frame
(`src/roomscope/eval/depth_bias.py`, `out/diag/depth_bias_*.txt`): median relative error -1.30 %,
-1.10 %, -1.23 % on 42444946/49/50, flat across range bins 0.5-4 m, consistent with Zea & Hanebeck
(JAIF 2022/23, "always negative, 1-2 % of range"). Floor and ceiling are both pulled toward a camera held
between them, so a 3 m ceiling shrinks by about 3.6 cm.

**Predicted after-number.** |ceiling error| <= 1 cm on the scored capture, and mean wall offset <= 2 cm
with drift off.

**Fix.** Divide depth by a scale fitted on *other* captures (default 0.9883 from 42444949 + 42444950;
`config/depth_scale.yaml`, `core/depth_calib.py`), applied at load time before drift correction.
`--no-depth-correction` restores the old behaviour for the ablation.

**Before / after (regenerable).** `python bench/fix_loop.py` re-runs the current pipeline on all three
captures with the fix off and on; the scale for each capture is fitted on the other two only.
Results: `fix/before_after.md`.

| capture | held-out scale | ceiling err cm | wall-plane MAE cm | wall-length MAE cm |
|---|---|---|---|---|
| 42444946 | 0.98835 | -3.23 -> **+0.15** | 0.91 -> 1.62 | 1.28 -> 1.80 |
| 42444949 | 0.98735 | -2.66 -> **+0.55** | 5.40 -> 3.49 | 9.63 -> 9.01 |
| 42444950 | 0.98800 | -3.82 -> **-0.76** | 3.69 -> 3.00 | 7.36 -> 6.41 |

**Prediction vs outcome.**
- Ceiling, predicted |err| <= 1 cm: **met on all three captures**, each scored with a scale it did not
  help fit (mean |err| 3.24 -> 0.49 cm).
- Walls, predicted mean offset <= 2 cm: **met only on 42444946**, and there the walls got slightly
  worse (0.91 -> 1.62 cm; the corrected run also found a seventh wall). On 42444949/50 the walls
  improved but stay above 2 cm because of the entry-alcove walls (16 and 11 cm off), which are barely
  observed; their error is not depth scale and the fix could not touch it. The wall half of the
  prediction was wrong: it assumed the 2-4 cm inward offset seen in an earlier pipeline version, which
  later wall-selection fixes (layered-surface rule) had already removed on 42444946.

## Fix 2: wall-length intervals are uselessly wide (calibration gate)

Declared and committed (tag `fix2-before`) before any fix code.

**Worst gate.** Calibrated intervals on every number, with overconfidence penalised. LiDAR wall-length
intervals are honest but useless: leave-one-room-out coverage 0.92 [0.81-0.98], median half-width
**19.7 cm** for a median error of 4.8 cm (width/error 4.1), mean interval score 65.2 cm (n=51 walls,
7 rooms, `config/calibration.json` at 56d25d9).

**Root cause (evidence, `bench/wall_audit.py`, `out/diag/wall_audit_lidar.json`).** A wall's length is
the distance between its two neighbouring walls, so its error is set by them, not by the wall itself.
Every wall with |error| >= 9 cm has exactly one neighbour 10-20 cm off while the other is within
~1 cm; 8 of those 11 bad neighbours were observed over <= 20 % of their length (inferred from the floor
extent, which furniture cuts short: 8/11 lie inside the true wall). Over all 51 walls, error vs the
smaller of the two neighbours' observed fractions: Spearman -0.53 (p = 5e-5); both neighbours >= 30 %
observed: median error 1.4 cm (n=31); otherwise 13.7 cm (n=20). One global conformal quantile is set
by the second group and applied to the first. (The wall's *own* observed fraction does not predict
error, Spearman 0.01: the earlier normaliser looked at the wrong wall.)

Rejected fix: placing inferred walls from the ceiling edge (research suggestion). Checked first: the
ceiling edge near those walls is missing or runs 40-60 cm past the wall through openings; it matched
the laser on 1 of 8. Not built.
*Correction (added after fix2-after):* that check applied the saved cloud's world-to-result transform
to points already in the result frame. Redone correctly over all 11 bad neighbour walls: the ceiling
edge is within 5 cm of the laser wall on 4/11, missing on 5/11, and 18-29 cm off on 2/11. The
rejection stands (4/11 is not a reliable placement rule), but the numbers above were wrong. The root
cause and the fix are unaffected: they come from the laser scorer, not from the saved cloud.

**Fix.** Mondrian (class-conditional) conformal on wall length with two bins, "corner-supported" (both
neighbours >= 30 % observed) and "inferred" (otherwise), each with its own room-pooled quantile; the
bin is computed from the result itself at run time, and inferred walls are labelled in the output.

**Predicted after-numbers (leave-one-room-out, same 51 walls).**
- corner-supported bin: median half-width **10-15 cm** (from 19.7), coverage >= 0.85;
- inferred bin: median half-width 20-30 cm, coverage >= 0.85;
- mean interval score over all walls: **<= 50 cm** (from 65.2).
Uncertainty: the supported bin still contains ~19 % of walls above 8 cm, so its 90 % quantile cannot
be small; this fix makes the intervals adaptive, it does not make the geometry more accurate.

**Outcome (commit tagged `fix2-after`; regenerate: `python bench/calibrate.py --tier lidar --from-records`).**

| | predicted | measured (LORO, 51 walls / 7 rooms) |
|---|---|---|
| supported bin median half-width | 10-15 cm | **15.7 cm** (19.7 before): **missed by 0.7 cm** |
| supported bin coverage | >= 0.85 | 0.94 [0.79-0.99]: met |
| inferred bin median half-width | 20-30 cm | 20.9 cm: met |
| inferred bin coverage | >= 0.85 | 0.95 [0.75-1.00]: met |
| mean interval score | <= 50 cm | **47.8 cm** (65.2 before): met |

Why the supported bin stayed wide: 6 of its 31 walls are still > 8 cm off, and because its walls have
small raw uncertainties their normalised scores are larger, so its quantile q rose (7.05 -> 8.63)
even though its median error is 1.4 cm. The prediction anticipated the tail but underestimated it.
Next lever, not part of this fix: find what those 6 walls share (the corner research's "wrong layer"
case: a neighbour fitted to a front surface while the dominant laser surface lies behind it).

## Fix 3: video and photo intervals never contain the truth (calibration gate)

Declared and committed (tag `fix3-before`) before any fix code. Gates from `bench/results/gates.md` at 9daf662.

**Why this gate.** It is the worst one on the benchmark set:
- **Video and photo coverage:** the calibrated intervals contain the laser value for 0 of 4 video
  measurements and 0 of 3 photo measurements. The nominal level is 0.9.
- **LiDAR coverage:** 13/15.
- **Other gates:** the video and photo accuracy gates (±3 %, ±8 %) and repeatability also fail. But every
  video and photo number is reported with a near-zero interval, so none of them says it is uncertain. The
  brief caps the score for confident garbage: a wrong number with a tight interval is worse than a wrong
  number with an honest one.

**Root cause (evidence).** The video and photo intervals are raw plane-fit statistics: `fitstat:v0`, about
±0.9 cm on a ceiling and ±8 cm on a wall. `config/calibration.json` has an entry for LiDAR only, so
`calibrate.apply` leaves these tiers raw and only adds a warning. Their real error is the metric scale of the
RGB reconstruction, a fully correlated relative error that a plane fit cannot see. On the set:
- ceilings −7.7, −12.6, −27.4, −28.1 cm (video) and −27.7, −10.0, +7.4 cm (photo), i.e. 3–12 %;
- wall offsets 10–100 cm.

**Fix.** Fit split-conformal calibration for the video and photo tiers (`bench/calibrate.py --tier video|photo`)
on laser-scored captures disjoint from the benchmark set:
- ARKitScenes rooms other than 421337: 42897521, 42897647, 42897501, 42897545, 42898811, 42898818;
- MuSHRoom vr_room and coffee_room (long and short).

Rooms in the benchmark set (421337 and honka) are excluded from calibration, including 42444950, which is room
421337. Uses the existing normalised score u = sqrt(σ_raw² + (1 cm)² + (0.5 % · value)²), so q scales the
relative term that dominates for these tiers. No pipeline or geometry change.

**Predicted after-numbers.**
- Benchmark set, video and photo coverage together: **≥ 6 of 7** measurements. Today 0 of 7.
- Leave-one-room-out coverage on the calibration rooms: ≥ 0.80 per tier.
- Median ceiling half-width: **5–15 % of the value** for video and photo, from ~0.3 % today.

Uncertainty: with ~8 calibration rooms and few clean walls per video/photo run, the quantile is coarse and its
coverage CI is wide.

This fix makes the intervals honest. It does not make video or photo measurements more accurate: the ±3 % /
±8 % accuracy gates and repeatability are predicted to stay where they are.

**Outcome (commit tagged `fix3-after`).**
- **Calibration:** fitted with `python bench/calibrate.py --tier video|photo` on the 10 disjoint calibration
  captures (8 rooms).
- **After numbers:** `python bench/gates.py recalibrate --tiers video,photo` re-applies the new calibration to
  the before-runs (`out/bench/before`), then `python bench/gates.py score`.
- **Why recalibrate instead of rerun:** the calibration is the fix's only changed input and the geometry code is
  unchanged, so the calibration was re-applied to the finished runs.
- **Check that this is equivalent:** one capture was fully rerun (video 42444946). It gives the same ceiling
  interval width, ±0.69 m vs ±0.69 m, and both contain the laser value. Its geometry differs by 1 cm on the
  ceiling and 23 cm on one wall, because COLMAP is not deterministic between runs.

| | predicted | measured |
|---|---|---|
| Video + photo intervals containing the laser value (benchmark set) | ≥ 6 of 7 | **7 of 7** (0 of 7 before): **met** |
| Leave-one-room-out coverage on the calibration rooms | ≥ 0.80 per tier | video ceilings 1.00 [0.54–1.00] (n=10, 8 rooms): met. Photo: **not computable** (one ceiling per room) |
| Median ceiling half-width | 5–15 % of the value | video **±25 %**, photo **±103 %**: **missed, much wider** |
| Accuracy gates (±3 % / ±8 %) and repeatability | unchanged | unchanged (video 0/4, photo 0/2 repeatability; walls not measurable) |

**Why the widths missed.** The 5–15 % prediction came from the benchmark's own errors (3–12 %). The calibration
rooms are harder:
- **Video ceilings:** median error 20 cm, so the 90 % quantile is about 25 %.
- **Photo ceilings:** median error 52 cm, with some ceilings almost entirely wrong. Eye-level photos rarely see
  the ceiling.

**Two changes made while shipping, both labelled in the output:**
1. **Walls borrow the ceiling's relative quantile.** With fewer than 5 laser-referenced walls (video 2,
   photo 0), walls use the ceiling's relative quantile (|error| ÷ value). The first version transferred the
   normalised quantile; its normaliser is the measurement's own fit noise (about 0.5 cm for a ceiling, 5 cm for a
   wall), so it produced a −1.5 to 7.3 m wall. Caught before any after-run and replaced.
2. **Lengths are floored at 0.**

**Result.** Video and photo now state their uncertainty correctly: before, they claimed ±1 cm and were wrong by
10–30 cm. Accuracy is unchanged, as predicted; photo intervals of ±100 % correctly say these photos cannot
measure ceiling height.

**Next fix:** metric scale from the RGB depth model, the root cause of those widths.

## Fix 4: the same room measured twice gives different walls (repeatability gate)

**Worst gate.** G.3 repeatability: wall lengths from two captures of one room within max(1 cm, 0.5 %).
It is the lowest-scoring gate on the benchmark set (`bench/results/gates.md`, commit `dc2fd2b`):

| tier | pair | walls within tolerance | median length difference | ceiling difference |
|---|---|---|---|---|
| LiDAR | 42444946 / 42444949 | **0/8** | 14.0 cm | 0.6 cm |
| LiDAR | honka_long / honka_short | 3/6 | 2.1 cm | 0.5 cm |
| video | both pairs | 0/4 | 12–96 cm | 1.3–4.6 cm |
| photo | both pairs | 0/3 | 35–105 cm | 17–27 cm |

**Scope.** This fix targets the LiDAR tier, 3/14 (21 %). The video and photo rows have a different
cause: per-capture metric scale from the depth model is off by 3–12 %, so lengths and ceilings disagree
together. No outline change can reach 0.5 % there.

**Root cause (evidence: per-wall audit of both pairs against the Faro laser).**
- **Depth and poses repeat.** The ceiling agrees to 0.5–0.6 cm within each pair. Every well-observed wall
  sits within ±2 cm of the laser in *both* captures.
- **The outlines do not.** 42444946 has 10 edges and 42444949 has 12. Of these, 2 and 4 are floor-boundary
  edges with no wall plane behind them (observed fraction 0): the visible floor stopped at furniture, not at
  a wall.
- **Each notch cuts a real wall into different pieces in each capture,** so the same wall gets different
  lengths. honka: one 0.42 m floor-boundary notch in honka_long (−19.8 cm vs laser) splits the long wall.
  It reads 5.39 m there and 5.83 m in honka_short, which has no notch. The same notch costs 3 of the 4
  head-to-head losses.

**Fix.** A floor-boundary edge (no wall plane, not a shared boundary with another room) is not a wall.
- When the notch is no deeper than the reach used to find a wall behind furniture (`STRUCT_REACH`, 0.8 m,
  wardrobe/counter depth), the edge is removed and its two neighbours merge. The better-supported wall
  plane keeps the position.
- Deeper floor-boundary edges stay, marked inferred (observed fraction 0, widened interval), as now.
- No other threshold changes.

**Predicted after-numbers (LiDAR, benchmark set).**
- G.3-lidar: **≥ 40 %** of matched walls within tolerance (from 21 %); honka pair ≥ 4 matched walls within.
- Floor-boundary edges on the four laser captures: from 7 to ≤ 3.
- Head-to-head vs Polycam: **≥ 70 %** (from 69 %): the honka notch's three losses go away.
- Do no harm: no laser-scored wall position more than 2 cm worse; ceiling and calibration gates unchanged.
