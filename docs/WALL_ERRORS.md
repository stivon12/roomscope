# Root cause of the 10–21 cm wall errors (2026-10-05)

Scope: every wall length or wall plane off by 10 cm or more against the laser on the current code: 33 numbers
on 9 captures (42444950, 42897501, 42897545, 42897647, 42898811, 42898818, MuSHRoom honka_long,
coffee_room_short, coffee_room_long). Three independent read-only investigations reran the geometry with
instrumentation (every candidate plane, which planes `drop_layered` removed, which plane each polygon edge
snapped to), compared per edge with the registered laser, and checked RGB. Scripts and images are kept
outside the repo.

## What it is not
- **Not measurement error.** Where our cloud and the laser see the same surface they agree within 0–3 cm.
  One edge out of all of them is a genuine 4 cm offset of the right surface (42444950 W7, cause open).
  MuSHRoom shows a separate, smaller ~4 cm outward bias on matched surfaces (Polycam depth or poses).
- **Not drift, depth scale or registration.** Errors are unchanged with drift correction off. Well-observed
  neighbouring walls score 0.1–3.6 cm. ICP fitness is the same as on the good captures.
- **Length errors are inherited.** Every bad wall length comes from one bad neighbouring edge, because a
  length is the distance between the two neighbouring wall positions.

## Root cause in the pipeline
The room boundary is never decided from structure. Three things combine:
1. **A wall hypothesis is one global line per facing direction.** It is one peak in a 1-D histogram over
   the whole capture, and its extent is made by bridging gaps up to 1.25 m. Where along the wall the surface
   actually exists never enters any decision. So:
   - separate surfaces merge into one blended plane (honka: a step face plus furniture fronts);
   - a nearby stronger wall suppresses a weaker one (42897545: the landing wall lost to the bathroom wall
     6 cm away);
   - a few stray points stretch a plane across the room (42898811: appliance sides became full-length
     cuts that carved a notch; coffee_room: a curtain plane stretched over the wall next to it).
2. **Layering is judged in 1-D.** `drop_layered` keeps the rear surface and drops the front one using only
   offsets and bridged overlap. At a step in a wall, or where blinds, glass or a curtain hang in a window
   recess, it deletes the real wall:
   - 42897501 W15: vertical blinds in the window recesses beat the splashback wall;
   - coffee_room short and long: the curtain plane beat the flat wall beside it;
   - honka: the step face was dropped together with the furniture fronts.

   Sometimes it keeps a curtain-plus-wall blend instead (42897647 W11).
3. **The polygon is the outline of the visible floor, snapped to any same-facing plane within 0.3 m.**
   - Nothing checks that the plane is supported along that edge; a plane seen only beyond another wall
     is accepted (42444950 W5, 42897545 R1-W4).
   - Nothing checks which side of a thin wall the room is on (42898818 W8: 9.4 cm, the wall's thickness).
   - When no plane is near, the edge **silently stays where the floor stopped being visible**: a counter,
     cabinet, bath, chair stack or curtain hem, 12–20 cm short of the wall.
   - `observed_fraction` still reads 0.7–0.95 there, because it counts any solid returns near the edge,
     so the interval does not widen. coffee_room short W3 (+21 cm) is the one uncovered miss.

## Root cause in the scorer (eval/laser.py)
The laser reference for an edge is "any laser surface facing the right way within ±20 cm of OUR edge".
That makes it a mirror image of the pipeline's flaw:
- when our edge is on furniture, the reference is furniture: a counter front, appliance side, sofa, bath
  rim, door leaf or entry cabinet (42898811 W5/W6, 42897501 W6/W8, 42897545 R2-W6, 42444950 W5);
- it hides real wrong-surface errors. On 42444950 W2 our edge sits on a wardrobe or curtain 28 cm in front
  of the wall and scores +0.8 cm. The reference lengths for the same physical wall differ by 29 cm between
  42444949 and 42444950;
- it misses a step that lies outside its window (coffee_room long W4: the real step is ~1 m away).

So part of the benchmark measures our output against itself. The wall numbers are not trustworthy until the
reference is extracted from the laser independently of our edges.

## Why semantic labels did not help
Only some failures are class confusions (blinds, curtain, counter, cabinet). Most are geometric: merged or
over-extended planes, layering judged in 1-D, and the floor-boundary fallback. The A/B on 15 captures
changed only 42444946. Labels can help *inside* a structural model (a curtain is never the wall), not as a
veto on top of the current one.

## Conceptual fix (not threshold changes)
1. **Scorer first,** or no fix can be measured. Extract the laser's structural walls on their own:
   - surfaces continuous from floor to ceiling along their length, around openings;
   - matched to our walls by layout, not by the nearest surface within a window;
   - an edge whose laser profile is bimodal along its length (a step) is flagged, not scored against one
     plane.
2. **Walls as supported segments, not global lines.** Each wall gets its own extent and height profile per
   stretch, built from evidence. Layering and occlusion are decided locally: a front surface is clutter only
   where the rear surface exists at the same along-wall position and the camera rays show it is hidden.
   Structural evidence is reaching the ceiling, continuity around openings, and no free space seen behind.
3. **No silent fallback.** An edge may snap only to a segment whose own support covers that edge. An edge
   with no wall evidence is marked inferred (`observed_fraction` 0, wide interval) or estimated from
   structure: the ceiling edge, or the same wall continuing. It is never reported at the floor boundary as
   if observed.
4. **Rooms from structure.** The room is the arrangement of structural walls that best explains the
   free-space evidence, rather than a snapped outline of where the floor was visible.

## Step 1 done: structural laser reference (eval/laser._structural_wall)
How the reference is now found for each of our polygon edges:
- **Search region.** The laser is searched ±1 m across the edge over its along-wall span. Our edge picks
  which wall is meant, not where its reference is.
- **Candidates.** Every room-facing laser surface is a candidate.
- **Structure.** A surface counts as structure if it reaches the ceiling above it (downward-facing returns
  above 1.8 m) over at least two 10 cm stretches. Counters, sills, blinds, sofas and anything seen through
  a window never do.
- **Occluders.** A structural surface in front of another is an occluder (wardrobe, curtain) when the
  rear one continues on both sides of it.
- **Levels.** Each remaining structural surface is a level. So is a tall surface behind the chosen level
  over stretches where that level is not seen.
- **Status:**
  - one level: `wall`, scored;
  - several levels: `step`, reported with the error to every level and the smallest error as a lower
    bound, not scored;
  - no structural surface: `unscored`, listed with the reason.

  Every edge lands in exactly one of these lists. The old scorer dropped edges silently.

Rules tried and rejected while building it (each checked on the edges where it went wrong):
- **Rearmost surface per stretch:** picks things seen through windows (42897501 W20).
- **Most-supported full-height surface:** picks a wardrobe over the wall beside it (42444949 W2).
- **Floor-to-door-height surface where wall tops were not scanned:** accepts a closed door in its reveal
  as the wall (42444946 W4).

**Check on the reference itself.** The three captures of visit 421337 share one laser frame. Every
reference line was mapped back into that frame and compared with the other captures:
- **The wall behind the wardrobe:** old reference 29.1 cm apart between 42444949 and 42444950; new 0.6 cm.
- **Clean `wall` pairs:** every one now agrees within 2.7 cm. The remaining 25 cm pairs compare different
  surfaces: the wall beside the wardrobe against the wardrobe front.

**What the old scorer hid** (examples confirmed in laser cross-sections):
- 42444946 W2: our edge sits on a cabinet front, 37 cm in front of the wall. The old scorer gave no
  reference.
- 42898811 W3: 41 cm short of a clean wall. No reference before.
- 42897501 W5: a phantom notch, 31 cm. It scored −1.0 cm against nothing.

**Wall accuracy on the same results** (pipeline unchanged; 2026-10-05, uncalibrated runs in out/calib/lidar
and out/mushroom):

| | old scorer | structural reference |
|---|---|---|
| ARKitScenes wall position | n=72, median 1.5 cm, max 19.6 cm | wall n=44: median 1.7, mean 10.0, max 60.3 cm, 12 > 10 cm; step n=16 (lower bound median 1.4); unscored 38/98 |
| ARKitScenes wall length | n=53, median 1.8 cm | wall n=21: median 8.7, mean 15.5, max 59.5 cm, 9 > 10 cm; step n=22 (lower bound median 3.7) |
| MuSHRoom wall position | n=48, median 1.8 cm, max 19.5 cm | wall n=27: median 2.7, mean 6.1, max 44.0 cm; step n=11; unscored 26/64 |
| MuSHRoom wall length | n=40, median 2.2 cm, max 21.3 cm | wall n=14: median 6.1, mean 14.0, max 43.1 cm; step n=12 |

The wall-length median of 1.4–1.8 cm reported before came from the scorer, not the pipeline.

**Consequences:**
- The conformal wall calibration (`config/calibration.json`) was fitted against the old reference and
  must be refit.
- `tests/test_lidar_real.py::test_wall_planes_within_noise_floor` is now a strict xfail.
- The unscored edges are mostly under 0.5 m, too short for three 10 cm stretches after trimming the ends.
  In MuSHRoom vr_room they are long walls whose tops the laser did not scan.
- `test_drift_correction_helps` is also a strict xfail. On 42444946 the drift-corrected polygon has 10 edges,
  one of them (R1-W2) on a cabinet front 37 cm in front of the wall. The uncorrected polygon has 7 edges and
  no such edge. Mean error on the scored walls: 9.8 cm with drift correction, 2.2 cm without. The drift
  residual itself still drops. Whether drift correction causes the cabinet edge is open (pipeline, step 2).
