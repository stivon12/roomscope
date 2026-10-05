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
