# Project review (2026-10-05)

Six read-only reviewers went through code, outputs, environment, evaluation and the brief. Five research
agents then looked for fixes to the non-bug problems. This page lists every finding with its evidence and a
command to check it. It also gives my verdict and the decision I propose, with the reason.

**Who checked what.** ✔ = I re-checked it myself after the reviewer reported it. R = reported by a reviewer
with quoted evidence, but not yet re-checked by me. Research claims marked *unverified* by the agents are
kept as such.

---

## 1. Where the project stands

| Area | Honest status |
|---|---|
| LiDAR tier (ARKitScenes) | Works on development data. Ceiling within 1.5 cm on 9/11 rooms-captures; walls well-measured where both neighbours are observed. The walk-in path (Stray Scanner on an iPhone) has never run on a real export and has a confirmed pose bug (§2.1). |
| Photo tier | Runs. Wall lengths within ±8 % on 3 rooms, but on stand-in photos with known intrinsics. Ceiling unreliable. Two bugs make real-iPhone input worse than the benchmark (§2.3, §2.4). No calibrated intervals. |
| Video tier | Not accurate. No working accurate path yet. |
| Damage, concealed flags, scope | Not built. These three output fields are always empty. |
| Own benchmark data | None collected. Every number comes from ARKitScenes iPad scans and Faro lasers. |
| Fix loop | Two fixes shipped with regenerable before/after. The brief wants the worst gate in *our own* benchmark, so they need re-doing on our data. |

**Score exposure, from the brief.** The largest exposures are:
- **Walk-in test, 30 %.** It must run on the reviewer's iPhone; all three real-device paths are untested.
- **The calibration cap.** "Confident garbage on thin input caps your total score", and photo/video intervals are uncalibrated.
- **Missing contract fields.** Damage, concealed flags and scope are empty.
- **Benchmark and head-to-head, 25 %.** These need our own captures.

---

## 2. Corrections to things I told you earlier

| What I said | What is true | Check |
|---|---|---|
| "Ceiling within 1 cm on 3/3 held-out captures" (README, COMPLIANCE) | Those 3 captures are all one room, and two of them helped fit the depth scale. Across all 11 ceilings, **2 fail the 1.5 cm gate**: 42898811 rooms R2 and R3 at −1.58 and −2.42 cm. ✔ | `jq '[.[]\|select(.kind=="ceiling_height")\|(.value-.truth)*100]' out/calib/records_lidar.json` |
| Wall-length statistics (coverage 0.92–0.95, median 1.4 cm) | They describe only the walls the scorer could find: **51 of 84** reported walls. Walls more than 20 cm off get no reference and are silently dropped. ✔ | §3.1 |
| SfM trajectory error 13.7 cm (and 41–63 cm); photo rotation errors; "ground-truth timing noise 3–10°" | All are invalid. My video-to-ARKit time mapping used `traj_start + i/fps`, but the video starts **0.633 s before** the trajectory. ✔ | §3.9 |
| Fix 2 "declared before any fix code" | The tag diff is exactly the fix, but the declaration and fix commits are 61 s apart. The code was drafted before the declaration commit. Predictions were written before the run (one missed), but git cannot prove that. R | `git log -1 --format=%ad fix2-before; git log -1 --format=%ad fix2-after` |
| Fix 2 root-cause statistics (Spearman −0.53, 1.4 vs 13.7 cm) | Correct when re-fitted, but **no committed script regenerates them**. They came from ad-hoc commands. R | `grep -rn -i spearman bench src` finds nothing |

---

## 3. Confirmed problems, by severity

### Critical

**3.1 The scorer drops every wall it cannot find within ±20 cm ✔.**
- **Where:** `eval/laser.py:345-350`, `:401`.
- **What happens:** 84 walls are reported and 51 scored. On 42897647, 1 of 16 walls is scored. All "bad" errors sit at 9.7–19.3 cm, just inside the window, which looks like truncation.
- **Effect:** every accuracy and coverage figure is flattered.
- **Decision:** rebuild the scorer.
  1. Extract the laser's own wall plan independently, using a slice at about 1.6–2.2 m, which avoids furniture.
  2. Freeze that plan once per venue.
  3. Match walls one to one (Hungarian assignment, gates on angle, offset and overlap).
  4. Count misses and phantoms.
- **Why:** this is how the ISPRS indoor benchmark (Khoshelham 2020) and RoomFormer/Floor-SP score. It also makes the ceiling reference the same for every capture, which removes the 1.4 cm swing.

**3.2 Stray Scanner poses are applied in the wrong camera axes ✔.**
- **Where:** `frontends/lidar.py` (`load_stray`).
- **What happens:** the pose comes straight from the quaternion as ARKit camera-to-world. Stray actually writes `q_WA * q_AC`, where `q_AC` is a 180° flip about x, so its poses are in OpenCV camera axes. I checked Stray's `OdometryEncoder.swift`. Our points are in ARKit axes (y up, −z forward), so every frame would be flipped 180° about the camera x-axis.
- **Effect:** the walk-in LiDAR path would produce garbage.
- **Decision:** multiply by `CV_TO_OURS`. Also:
  - use the per-frame intrinsics in `odometry.csv` columns fx, fy, cx, cy, scaled by the real `rgb.mp4` size, instead of `camera_matrix.csv` and the "width = 2·cx" guess;
  - join on the `frame` column, because depth PNGs can be missing;
  - process every recording in the folder;
  - subsample to about 10 Hz.
- **Verification needs a real Stray export.** Until then, check against StrayVisualizer's own reader.

**3.3 Fusion uses every second view ✔.**
- **Where:** `core/layout.py:94` and `:762` default to `frame_step=2`; `pipeline.py:38,72` never override it.
- **Effect:** with 8 photos only 4 contribute points, free space, wall observation and door evidence. Video uses 16 of 32 keyframes. This likely contributes to the photo floor patches and to zero detected doors.
- **Decision:** set the step from the tier: 1 for photo and video, a time-based subsample for LiDAR.

**3.4 Real iPhone photos break the known-camera path R.**
- **HEIC conversion drops EXIF.** `photo.py:39` saves without `exif=`, so no focal length reaches the model and it guesses. The research found MapAnything's guess was 34 % low.
- **Portrait JPEGs carry EXIF Orientation 6.** `reray_known_K` (`recon.py:132`) uses the un-rotated size, which gives a 1.32× focal error and an 86 px principal-point shift.
- **Second-run duplicates.** A second run sees both `X.HEIC` and the `X.jpg` written by the first run, and uses both, so every view is duplicated.
- **Why the benchmark hides this:** stand-in photos have no orientation tag and carry an exact-K sidecar.
- **Decision:**
  - read EXIF from the HEIC and keep it, or skip conversion;
  - call `exif_transpose` before taking image sizes;
  - write converted files and caches under `out/`, never into the user's folder;
  - deduplicate by file stem.

**3.5 Damage, concealed flags and scope are not built R.**
- **Where:** `damage/` and `scope/` are empty and `rules/` has only `.gitkeep`.
- **Decision:** the design is in §5.4.

**3.6 Openings: zero doors ever detected R.**
- **What happens:** 15 openings across 9 captures, all typed "window", several with door geometry (sill 0.16 m, top 2.04 m).
- **Cause:** a door needs see-through evidence in the bottom 15 cm, and a camera at 1.4 m almost never gets rays there. So adjacency between rooms is always empty. Openings are also reported on walls with 0 % observed, and their intervals claim a 90 % level with no ground truth.
- **Decision:** the rule fix is in §5.2. Label opening intervals "uncalibrated" until opening ground truth exists.

**3.7 No benchmark data of our own R.** The brief: "you build the benchmark set yourself … so it cannot be flattered". ARKitScenes is an iPad Pro 2020, not an iPhone 15+, and it lacks a staged damage room, a multi-room house captured at all tiers, and a Polycam comparison. **Needs you** (§6).

### High

**3.8 Photo and video intervals are raw fit statistics labelled 90 % R.**
- `config/calibration.json` has LiDAR only.
- When no ceiling is seen, the 2.4 m default is reported as a measurement (±8 cm). One photo room reported a 0.99 m ceiling with no warning (the room was reconstructed in two pieces 1.1 m apart vertically).
- **Decision:**
  - label defaults as priors with wide intervals (2.44 m [2.29, 2.74]);
  - reject ceilings below 2.0 m;
  - fit photo calibration once the scorer is scale-aware (§5.3);
  - until then, intervals no narrower than the gate (±8 % photo, ±3 % video).

**3.9 Video-to-trajectory timing is off by 0.633 s ✔.** The `.mov` (5644 frames at 60.02 fps) matches the 5644 `lowres_wide_intrinsics/*.pincam` files one to one, and the first pincam is 0.633 s before the first `lowres_wide.traj` sample.
- **Decision:** frame *i* gets the time of pincam *i*. Interpolate poses only between bracketing samples. Then re-measure every SfM and photo pose comparison.

**3.10 The calibration records mix pipeline versions R.** `bench/calibrate.py` reuses cached results unless `--rerun`. The 9 cached LiDAR results come from different commits, and results carry no commit stamp.
- **Decision:** stamp `git rev-parse HEAD` (marked dirty when relevant) into `meta`, then re-run everything after the bug fixes.

**3.11 The iPad depth correction is applied to iPhones R.**
- **What happens:** with no `--device`, the −1.17 % scale fitted on an iPad Pro 2020 is applied, and its uncertainty never enters any interval.
- **Research:** the published iPad bias is 1–2 % of range. Apple changed its depth processing on the iPhone 15 Pro, and no per-model scale is published.
- **Decision:** unknown devices default to 1.0 with a printed sensitivity (±1.5 %), and the scale uncertainty enters the interval.

**3.12 The opt-in video SfM still runs the mapper that failed R.** The worker defaults to the global mapper; the "structure-less off" fix applies only to the incremental one. Quadratic sequential matching gives 0 matches beyond a gap of 16 frames.

**3.13 Environment drift R.**
- Two OpenCV builds (5.0 and headless 4.10) share one `cv2/` folder.
- `vggt` (unused) pins numpy < 2.
- `pillow-heif` and `mapanything` are not declared in `pyproject.toml`.
- There is no lock file, and weights load without a pinned revision.
- **Decision:**
  - keep headless OpenCV only;
  - uninstall vggt;
  - declare the missing dependencies;
  - commit `uv.lock`;
  - pin the model revision and record its hash;
  - pre-fetch the torch-hub DINOv2 code in `fetch_weights.sh`.

**3.14 The photo tier reloads the 5 GB model for every room R.**
- **Decision:** load it once per property.

**3.15 Stand-in photos are not protocol photos R.** They are picked by viewing direction from the video, mostly pitched down, rarely show the ceiling line, and include no doorway photos, which the protocol requires.
- **Decision:** pick frames near room corners with both floor and ceiling lines in view, add doorway frames, and also run with focal length derived from the 35 mm-equivalent value rather than exact K.

**3.16 Smaller confirmed issues R.**
- `meta` drift residuals are not comparable before vs after: every capture shows about 40 % "improvement" while the laser shows none.
- A capture that finds no room writes schema-invalid JSON (`rooms` minItems 1).
- One failing room folder aborts the whole property.
- Live Photo `.MOV` companions make a photo folder look like video.
- The protocol promises a low-light flag that does not exist.
- Video frame extraction seeks before every candidate frame, about 205 s of a 392 s run.

### Medium / low R
- **Multi-room scoring frame:** the cloud of multi-room photo captures is saved in the first room's frame, not the stitched one.
- **Footprint area:** it is the sum of room areas rather than the union.
- **Single-axis-set assumption:** one Manhattan frame and one floor plane mean angled walls, sunken rooms and steps disappear without a warning.
- **Coverage intervals:** Clopper-Pearson over walls treats them as independent, but errors cluster in 7 rooms. Report per-room results and a room-level interval.
- **Cache key:** the MapAnything cache ignores K values, model revision and threshold.
- **Tests:** `tests/` needs 43 GB of data, so it skips on a clean clone. Two assertions are tautological. There are no unit tests for calibration or tier detection.
- **Dead code:** the self-calibrating depth estimator and unused drift options.

### Checked and solid
- Git hygiene: no data, weights or outputs are tracked, and history is incremental.
- Fix 2's numbers reproduce exactly from records, and the tag diff is exactly the fix. Fix 1 is regenerable and leave-one-capture-out.
- Room-grouped leave-one-room-out calibration, with labelled small-n fallbacks.
- ARKitScenes pose conventions; confidence-2 filtering; the see-over furniture test.
- MapAnything crop maths for EXIF-free images; COLMAP isolated in a subprocess.
- Video `K` derivation for ARKitScenes is correct. The "uncalibrated" frame pairs come from weak matches, not a camera mismatch.

---

## 4. Decisions already made, and why (for review)

| Decision | Why | Evidence |
|---|---|---|
| LiDAR depth ÷ 0.9883 by default | Per-pixel comparison with laser-rendered depth: −1.1 to −1.3 % | `out/diag/depth_bias_*.txt`, fix 1 |
| Drift correction defaults to "do no harm" | Earlier settings made poses worse against the laser (2.4 → 8.2 cm); every stronger variant hurt at least one capture | `out/diag/drift_*.txt` |
| Wall intervals split by neighbour support | Wall length depends on its two neighbours | fix 2 |
| MapAnything for photo/video, not VGGT | VGGT is only marginally better (3.6° vs 4.9°), non-commercial and slow | RESEARCH §8 |
| Video SfM opt-in only | Never reached a consistent reconstruction | §3.12 |

---

## 5. Research: proposed approach for each open problem

**5.1 Photo and video geometry.**
- **The finding:**
  - MapAnything treats given intrinsics only as conditioning; the maintainer says the model may change them (issue #150).
  - Its paper: scale error about 13 % from images plus K, **about 5 % when camera poses are also given**, about 2 % with depth.
  - The repo ships `scripts/demo_inference_on_colmap_outputs.py`: COLMAP intrinsics and poses in, metric scale and dense depth out.
- **Proposed video pipeline:**
  1. Keyframes by optical-flow parallax (about 30–50 px), not equal time bins.
  2. DISK + LightGlue (Apache) for matching, in a torch-only process. Do not import hloc in that process: it imports pycolmap.
  3. COLMAP incremental mapping with K fixed, in a pycolmap-only process.
  4. MapAnything with those poses.
  5. A second scale check from a metric depth model (DA3METRIC-LARGE Apache, or Depth Pro with known focal).
  - **Fallback:** VGGT-Long chunking with MapAnything (`using_sim3: False`). Not tested on MPS.
- **Proposed photo pipeline:** the same scale anchors plus a door-height cue (2.03 m US / about 1.98–2.04 m EU, roughly ±2–3 %). Detect and re-run views whose floor height disagrees with the rest.
- **Ceiling from photos:** segment wall, floor and ceiling (ADE20K SegFormer), then use the camera-height ratio `H = h_cam·(1 + tanβ/tanα)`. Scale cancels, so MapAnything's scale error doesn't touch it. Count points as ceiling only where segmentation says ceiling. Add a level "height shot" per room to the protocol.
- **Honest expectation:** video ±3 % is not demonstrated anywhere for RGB-only capture. Photo ±8 % looks reachable once scale is anchored (*unverified*).

**5.2 Openings.**
- **Rule fix, about 1 day.**
  - A door is see-through anywhere in 0.3–1.8 m with no solid evidence below, a top at 1.95–2.15 m and a solid band (lintel) above.
  - Widths 0.6–1.0 m are a door; wider than 1.5 m is an open passage.
  - A window has solid wall below a sill at 0.4 m or higher.
  - Require the wall to be at least 60 % observed and both jambs observed.
- **Detector and fusion, later.**
  - SAM 3 through Hugging Face transformers runs on Apple Silicon (gated download, SAM licence).
  - Grounding DINO tiny is the fallback.
  - Vote on the wall plane, then refine each jamb from 1920×1440 RGB edges, because the RGB pixel at 2 m is about 1.5 mm versus about 1.3 cm for depth.
- **Ground truth:** ARKitScenes has no door or window labels, so openings must be annotated on the Faro scans, about 2–5 min each.
- **Expectation:** no published phone method shows ≤2 cm on 85 % of openings. LiDAR might reach 50–75 % (*unverified*).

**5.3 Evaluation.**
- **Scorer:** the redesign in §3.1.
- **Photo and video registration:** gravity-constrained 4-DoF plus scale, i.e. 2D Umeyama on wall lines, or Open3D point-to-point ICP with `with_scaling=True`. Report scale error separately from shape error.
- **Ceiling reference:** a whole-room ceiling plane from the laser, not one under our polygon.
- **Reporting:** a per-room table, and a room-level interval with the effective n stated.
- **Ground truth for poses:** `lowres_wide.traj` is ARKit's own tracking, not laser-refined, so it is only good to about 2 cm.

**5.4 Damage, concealed flags and scope (about 1 day for a minimal version).**
- **Detection:**
  1. Claude vision classifies each keyframe and returns pixel boxes.
  2. SAM 2.1 refines the masks.
  3. Masks are projected onto the fitted planes on a 1 cm grid.
  4. Areas get erode/dilate plus pose-jitter intervals.
  5. Reflections are rejected when the mask does not land on the same plane cells from every view. Low light is marked "insufficient evidence".
- **Rules:** a rule table with public citations (HUD flood-cut heights, EPA mold size tiers). IICRC standards are paywalled, so only their concepts are cited.
- **Scope:** our own code namespace with Xactimate category hints and no prices.
- **Replay:** a deterministic cache keyed by hash of image bytes, prompt, model and settings.
- **API constraints reported by the research (to re-check before building):** Sonnet 5.5 rejects non-default `temperature`, and rejects forced tool calls. If so, structured JSON output is the way, and only the cache gives determinism.

**5.5 Real devices.**
- **Stray Scanner:** the format and pose fixes are in §3.2. Recordings run at 60/30/15/5/1 Hz, and depth is `sceneDepth`.
- **iPhone videos:** the focal length is in QuickTime `mdta` metadata (`com.apple.quicktime.camera.focal_length.35mm_equivalent`). It can be read with about 50 lines of standard-library box parsing.
- **Protocol changes:** HDR Video off, Enhanced Stabilization off, no Action or Cinematic mode, 4:3, the 1× (24 mm) lens and the 24 mm default lens, Stray at 15 or 30 Hz, transfer originals by AirDrop or cable.
- **Polycam free tier:** GLTF export only; dimensions and floor plans need Pro (*unverified*).

---

## 6. Proposed order of work

| # | Work | Who | Est. | Why first |
|---|---|---|---|---|
| 1 | Walk-in bugs: Stray axes/intrinsics/recordings/sampling (3.2), `frame_step` (3.3), HEIC and orientation (3.4), per-room failure handling, schema-valid empty result, Live Photo detection | me | 3–4 h | Protects the 30 % walk-in; all confirmed bugs |
| 2 | Environment: one OpenCV, remove vggt, declare deps, lock file, pinned weights; time a clean install | me | 1–2 h | Reproducibility and the 15-minute README target |
| 3 | Honest numbers: fix README/COMPLIANCE claims (§2), stamp commits, re-run calibration, script the fix-2 statistics, label defaults and openings as uncalibrated | me | 2 h | Avoids overclaiming in the live defence |
| 4 | Scorer redesign (§3.1, §5.3), then re-score everything | me | 1 day | Every later number depends on it |
| 5 | Damage, concealed flags, scope (§5.4) | me | 1 day | Three empty contract fields |
| 6 | Door rule fix (§5.2) | me | 1 day | Adjacency and the openings gate |
| 7 | Own benchmark capture: ≥3 rooms + connector at all tiers, one room twice per tier, staged damage (2 classes), tape measurements, Polycam on 2 rooms, an iPhone 15 Pro | **you** | 3–4 h | Unlocks benchmark (15 %), head-to-head (10 %), a fix loop on our data |
| 8 | Photo scale and ceiling (§5.1), video COLMAP+MapAnything (§5.1), calibration for photo and video | me | 2–4 days | Hardest; least certain |

Optional: ScanNet++ access (academic form) gives laser truth for doors.

---

## 7. How to verify the key claims yourself

```bash
cd ~/roomscope
# 2/11 ceilings fail; 51 of 84 walls scored
.venv/bin/python -c "
import json,glob; R=json.load(open('out/calib/records_lidar.json'))
print([round((r['value']-r['truth'])*100,2) for r in R if r['kind']=='ceiling_height'])
print(sum(len(rm['walls']) for f in glob.glob('out/calib/lidar/*/result.json') for rm in json.load(open(f))['rooms']), 'walls reported;', sum(r['kind']=='wall_length' for r in R), 'scored')"
# frame_step=2 never overridden
grep -n "frame_step" src/roomscope/core/layout.py; grep -n "L.fuse\|opening_evidence(" src/roomscope/pipeline.py
# Stray pose axes: Stray writes q_WA * q_AC (180 deg about x); our loader uses it as ARKit camera-to-world
curl -s https://raw.githubusercontent.com/strayrobots/scanner/main/StrayScanner/Helpers/OdometryEncoder.swift | grep -n "q_AC"
grep -n "ARKIT_TO_ZUP @ Ta" src/roomscope/frontends/lidar.py
# 0.633 s video/trajectory offset
D=benchmark/raw/arkitscenes/raw/Validation/42444946
awk 'NR==1{print $1}' $D/lowres_wide.traj; ls $D/lowres_wide_intrinsics | head -1
# fix-2 timing
git log -1 --format='%h %ad' --date=iso fix2-before; git log -1 --format='%h %ad' --date=iso fix2-after
# HEIC EXIF loss and orientation size
grep -n "convert(\"RGB\").save" src/roomscope/frontends/photo.py; grep -n "Image.open(p).size" src/roomscope/frontends/recon.py
```

## 7. MapAnything → Depth Anything 3 (2026-10-05)

Default reconstruction for the video and photo tiers is now Depth Anything 3. It uses DA3-BASE (0.12 B) for multi-view and pose-conditioned depth, and DA3METRIC-LARGE (0.35 B) for metric scale. Both are Apache-2.0 and run on the Mac GPU (MPS) in `.venv-da3`. Setting `ROOMSCOPE_RECON=mapanything` restores the old path.

The figures below come from these commands, run on 2026-10-05:
- `bench/video_eval.py`, writing `out/video_eval/summary_{mapanything,da3}.json`;
- `bench/calibrate.py --tier photo --rerun`, writing `out/calib/records_photo{_mapanything,}.json`.

**Video** (COLMAP poses, 32 frames). Scale error is measured against ARKit; ceiling and wall-plane errors against the laser scans.

| Capture | Scale MA → DA3 | Ceiling cm MA → DA3 | Wall-plane median cm MA → DA3 |
|---|---|---|---|
| 42444946 | −9.9 → **−2.7 %** | −37.9 → **−6.1** | **3.8** → 11.6 |
| 42444949 | +22.8 → **−6.8 %** | +27.1 → **−18.2** | 10.3 → **7.8** |
| 42444950 | +33.4 → **−5.5 %** | −164.6 → **−12.2** | 12.9 → **3.9** |
| 42897501 | +19.5 → **+2.4 %** | +28.7 → **+7.3** | 9.8 → **1.4** |

- DA3 takes 26 s per capture. The whole 4-capture run took 3 min 30 s with cached COLMAP.
- MapAnything was not timed on the same frames; that run was stopped at the user's request.
- Gates still failing: ±3 % scale is met on 2 of 4 captures, and the 1.5 cm ceiling gate on 0 of 4. DA3METRIC alone reads short on every capture (`bench/scale_cues.py`: −10.7, −13.7, −13.4, −2.3 %).

**Photo** (stand-in photos, 9 captures; 3 min 52 s for the whole run with DA3).

| | MapAnything (older run, before the empty-footprint fix) | DA3 |
|---|---|---|
| Captures producing a room | 4 of 9 | 7 of 9 |
| Walls scored / within ±8 % | 6 / 6 (median 3.2 %) | 20 / 14 (median 4.0 %) |
| Ceiling errors cm | −33.1, +75.8, +0.5, −144.7 | −29.7, −12.0, −31.6, +6.4, −0.7, −122.5, −33.7 |

- The two DA3 photo failures are 42444946 and 42897647. In both, the floor region is not a closed polygon, so it is dropped. On 42897647 the camera heights spread 0.55 m, so DA3-BASE's unposed poses are less consistent there.
- The photo comparison is not like for like. The MapAnything records predate later fixes, so a fair MapAnything photo re-run is still owed.
