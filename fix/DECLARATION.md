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
