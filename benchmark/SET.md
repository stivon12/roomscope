# Benchmark set (real data only)

Five real captures, each run at all three tiers: the smallest set that covers the brief's list.

| Capture | Device | Role | Ground truth |
|---|---|---|---|
| YC c7d28f72c6 | iPhone Pro, Stray | multi-room capture: 6 spaces with a hallway connector; photos as per-room folders | none |
| ARKitScenes 42444946, 42444949 | iPad Pro 2020 | one room captured twice (repeatability) | Faro laser |
| MuSHRoom honka long, short | iPhone 12 Pro Max | a furnished room captured twice | Faro laser |

It is defined in `bench/set.yaml` and scored by `python bench/gates.py run && python bench/gates.py score`,
which writes `bench/results/gates.md`. Real recordings and real photos only: nothing synthetic, no painted-in
damage.

## Against the brief's composition list

| Requirement | Status | What covers it | What is missing |
|---|---|---|---|
| One multi-room capture, three or more rooms plus a connector | **Partly met** | YC apartment, iPhone Pro, Stray: c7d28f72c6. Six spaces: office, dining room, living room, two bathrooms, and a hallway with the stairwell as the connector | No ground truth yet: shows the pipeline runs and stitches; accuracy unscored |
| One furnished room with staged damage in two damage classes | **Not met** | Frame-level check on 33 real damage photos from Wikimedia Commons (`benchmark/raw/damage_photos`, `SOURCES.csv`) | A physical capture of a room with staged damage, plus tape measurements of each damage region |
| The same rooms captured at all three input tiers, including the multi-room set | **Met, with a caveat** | All five captures at LiDAR, video and photo. The photo tier for YC arrives as per-room folders (`benchmark/photo_select/c7d28f72c6.yaml`) | The video and photo inputs come from the same recording as the LiDAR capture: video is its RGB stream, photos are stills from it. They are not separate walk-throughs with a camera app |
| At least one room captured twice at the same tier | **Met** | ARKitScenes room 421337 (42444946, 42444949) and MuSHRoom honka (long, short), each twice at every tier | |
| Laser or tape ground truth on everything | **Partly met** | Faro laser for both laser rooms (4 captures) | No ground truth for YC (multi-room); no opening ground truth anywhere; no damage ground truth |

## Where the data comes from
- **ARKitScenes**, Validation split: iPad Pro 2020 LiDAR, with the Faro laser of each visit. See `benchmark/raw/MANIFEST.md`.
- **MuSHRoom** (CC BY 4.0): iPhone 12 Pro Max through Polycam, with a Faro Focus laser per room. Converted by
  `bench/mushroom_to_stray.py`.
- **YC:** our own iPhone Pro Stray recordings of one apartment.
- **Damage photos:** Wikimedia Commons, CC0, public domain, CC BY or CC BY-SA. Licences and authors are in `SOURCES.csv`.

## How the tier inputs are made
- **Video:** the capture's RGB stream, meaning the ARKitScenes `.mov` or the Stray `rgb.mp4`.
- **Photo, single rooms:** `bench/make_photos.py`, following the protocol (sharpest frames spread around in
  heading). Poses only choose the frames; the photo tier never sees them.
- **Photo, the YC apartment:** chosen by hand from contact sheets, as a photographer standing in each room would
  shoot it (automatic selection from the LiDAR room split was rejected: it picked doorway shots of the next room
  and missed the office desk).

## Next: completing the set
Each item needs a physical visit:
1. **One damaged room:** a furnished room with two staged damage classes, recorded with Stray at LiDAR, with
   ordinary video, and as 2–8 photos.
2. **The YC apartment, measured:** a laser measure or tape on each room's wall lengths and ceiling heights, and on
   each opening, written to `benchmark/gt/<room>.yaml` (`TEMPLATE.yaml`).
3. **Separate tier recordings:** video and photo captures made apart from the LiDAR recording, so tier errors are
   independent.
