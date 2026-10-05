# Room segmentation: the conceptual problem (2026-10-05)

## Symptom
- **Whole-apartment scans:** on the first real iPhone multi-room scans (`benchmark/raw/yc/`), the whole apartment came out as **one room**.
- **First fix, too few splits:** a local "doorway rule" split the apartment into 5–6 plausible rooms.
- **Same fix, too many splits:** the same rule split the ARKitScenes **single-room** scans into extra 1–2 m² "rooms". Laser wall error went from 4.78 to 5.58 cm.

Each threshold change moved errors from one set to the other. That is the sign of a wrong formulation, not a wrong threshold.

## How rooms are found today (`core/layout.py: segment_rooms`)
1. A 2-D floor grid of observed floor plus free space from camera rays.
2. Cut the grid along every fitted wall line (extended across unobserved stretches).
3. Each connected component the camera visited is a room.

Rooms separate **only if every connection between them is cut by a wall line**. A doorway is, by definition, a hole in that wall line, so any two rooms joined by a door merge. The original code even deletes wall-line pieces where the camera walked through ("nobody walks through a wall"), which guarantees the merge.

## Root cause
**What a room is, is a global question, but the code answers it with local yes/no decisions.**
- A "room" is a region bounded by walls that connects to other regions through **narrow passages**.
- The current method can only separate regions with a continuous barrier. The doorway patch then tries to decide, gap by gap, from local evidence (gap width, jamb height, header points above it), whether the gap is a door.

Local evidence is ambiguous both ways:
- **Things that look like doors but aren't** (single rooms, false splits): gaps between a wardrobe and a wall, alcoves, kitchen niches, closet openings, furniture fronts fitted as walls, and the camera path weaving between them.
- **Doors that don't look like doors** (apartments, missed splits): a floor-only scan sees walls only to about 1.3 m, so there is no header and no jamb height; doors next to corners; open door leaves narrowing the gap; doorways seen from one side only.

Every rule trades one failure for the other. There is no threshold that separates them, because the decision ignores context: what is on each side of the gap, how big and how enclosed those regions are, and how strongly they are separated from each other overall.

## Contributing problems
- **No ground truth for room partitions.** ARKitScenes captures are mostly single rooms (they only test against over-splitting), and the apartment scans have no measured partition. Any method needs a scored check for both failure directions.
- **Wall lines come from noisy per-face histograms.** Furniture and built-ins can be fitted as walls (`kind="wall"`), so cuts already contain false barriers.
- **The cue has to be device- and tier-independent.** The walk-in can be any iPhone, with LiDAR or video (photos are one folder per room by protocol, so they don't need segmentation).

## What a correct formulation needs (questions for research)
- A **global partition** of free space into rooms: e.g. visibility-based clustering (points in the same room see each other; a doorway is a visibility bottleneck), distance-transform / Voronoi / morphological methods from robot room segmentation, or graph cuts.
- Robustness to partial scans (floor-only, low walls, one-sided doorways) and to furniture.
- Explicit uncertainty: when the partition is ambiguous, say so rather than guess.
- An evaluation that scores both over-splitting and under-splitting.

Research brief and findings: see "Research" below (filled in by the research agent's report).

## Research (research agent, 2026-10-05; items marked unverified were not checked against full text)

**Recommendation: structural-clearance persistence (SCP).** This replaces the per-gap doorway rule with a global partition:
1. An obstacle map from **wall-face points 1.0–1.3 m above the floor**, plus wall lines only where observed. Furniture under 1 m drops out, and the band is visible even in floor-only scans.
2. A free mask from observed floor plus ray-traced free space, minus obstacles. Unknown cells count as obstacles.
3. Clearance D = distance transform of the free mask; L = log D.
4. 0-dimensional persistence of the superlevel sets of L. Each region is born at a clearance maximum r_b (room centre) and dies at a saddle r_s (half a door width). Persistence = log(r_b / r_s) is **scale-free**:
   - bedroom ≈ 3.7;
   - bathroom ≈ 2.3;
   - L-shaped room or alcove ≈ 1.0–1.4;
   - wardrobe gap: no maximum at all.
5. Threshold τ at the largest persistence gap inside a prior band of about [1.5, 2.5]. The band comes from door and room size statistics; it is calibrated on external data (Bormann maps, Matterport3D) and **frozen before touching our captures**.
6. Watershed from the persistent maxima. Boundaries fall on bottlenecks, are snapped to wall lines and handed to the existing `room_polygon`.
7. Regions no frame observed are unknown, not rooms.

**Uncertainty:**
- each boundary's margin (persistence − τ) is mapped to a split probability;
- a boundary is "ambiguous" near τ, or when the saddle touches unknown cells (a one-sided doorway);
- when a boundary is ambiguous, both partitions are reported.

**Known failure modes:**
- open-plan spaces with arches wider than 1.5 m merge;
- tall freestanding furniture mid-room can create a false bottleneck;
- corridors with many doors may be absorbed into one room or broken up;
- closets merge into their room.

**Optional tie-breaker:** ray co-visibility across an ambiguous boundary (high co-visibility means merge).

**Fallback:** visibility spectral clustering. Cells are the wall-line arrangement, affinity is ray co-visibility from our frames, then a normalised cut with k chosen at the eigengap (the Mura 2014 / Ambrus 2017 formulation with our real rays).

**Ruled out:**
- **Learned models.** RoomFormer: CUDA, data with non-commercial licences, trained on complete scans. HEAT: GPL, non-commercial. Floor-SP and MonteFloor: CUDA. SpatialLM: outputs no rooms; CUDA. RoomPlan: rooms come from the scanning protocol (the user ends each room); iOS only.
- **The trajectory should not cut or uncut walls**, because people scan rooms from doorways.

**Evaluation:**
- pixel recall/precision (Bormann 2016);
- room P/R/F1 at IoU > 0.5;
- explicit over- and under-segmentation counts.

**Data:**

| Dataset | Licence | Use |
|---|---|---|
| Bormann's 20 furnished maps | non-commercial | published baselines, e.g. Voronoi 86.6/94.5 recall/precision |
| Matterport3D region polygons | academic terms of use, must be requested | real test with depth and poses |
| Structured3D | non-commercial | scale; can simulate floor-only scans |
| ARKitScenes | — | false-split test only |

**Key sources:**
- Bormann et al. ICRA 2016: https://whiteoak.umd.edu/roswiki/attachments/ipa_room_segmentation/article.pdf
- Hydra RSS 2022: https://ar5iv.labs.arxiv.org/html/2201.13360
- Turner & Zakhor 2014: https://www.scitepress.org/Papers/2014/46803/46803.pdf
- Mura et al. 2014: https://www.crs4.it/vic/data/papers/cag2014-indoor_architectural_reconstruction.pdf
- Ambrus et al. RA-L 2017: https://people.csail.mit.edu/sclaici/pdf/ambrusclaici2017room.pdf
- ROSE² (GPLv3): https://arxiv.org/abs/2203.03519
- RoomFormer: https://github.com/ywyue/RoomFormer
- RoomPlan CapturedStructure: https://developer.apple.com/documentation/roomplan/capturedstructure.md
- Unverified: the ToMATo gap-selection citation (Chazal et al. 2013); Ochmann 2016, Ikehata 2015 and Armeni 2016 details.

## Decision status (updated 2026-10-05)
- **SCP is implemented** (`core/rooms_scp.py`, commits 2254c4e, b88b3c1). The threshold band was frozen on the Bormann maps before use on our captures: furnished recall/precision 85.6 / 95.0 %.
- **Rooms kept only if entered:** a region the camera never walked into (seen through a doorway) is reported in the warnings, not as a room (Turner & Zakhor 2014).
- **The rooms form a consistent partition** (1d34886). Every step after SCP used to treat each room on its own, and room outlines overlapped by up to 2.3 m². Now:
  - contested unseen floor goes to the nearest room;
  - an edge facing another room's floor is a shared open boundary, never snapped to a wall;
  - any remaining overlap is reconciled jointly, and crossing wall faces are reported as drift.
- **Open boundaries become openings and adjacency** (0917f33).
- **42897545 splitting in two is correct, not a false split.** The capture starts on carpet in an adjacent room and ends in a tiled bathroom (video frames checked). ARKitScenes calls it one room because the scan centres on the bathroom.
- **Known failure, open: corridors.** A corridor is narrow everywhere: its clearance maximum (half its width, about 0.5 m) is below r_min, and its ratio to the door saddle (about 1.3) is below the band. So SCP gives it no seed, and it is absorbed into a neighbouring room (c7d28f72c6: the 1.1 m corridor joins Room 1). This is a property of the clearance-persistence cue, not a threshold issue. A corridor is defined by its shape (a long ridge of near-constant clearance), so it needs a medial-axis cue. Bormann's ground truth labels corridors as rooms, so a fix can be measured there before it touches our captures.
