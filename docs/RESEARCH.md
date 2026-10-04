# Research pass on the conceptual blockers (2026-10-04)

One literature search per blocker, then each proposed fix was checked against our own numbers before
anything was built. "Verdict" says whether the fix was adopted, and why.

## 1. Drift correction (gate G.4)

**Literature.** Choi, Zhou & Koltun, *Robust Reconstruction of Indoor Scenes* (CVPR 2015): fragment pose
graph, odometry edges between consecutive fragments, loop edges switched off by a line process / robust
kernel. Manhattan / structural-regularity SLAM: only dominant structural planes are landmarks.
Ablation practice: injected-drift sweeps, "do no harm" at zero drift, accuracy shown next to
self-consistency.

**Our failure, explained.** Self-consistency improved while accuracy got worse (2.4 -> 8.2 cm) because the
solver could explain sensor error and wrong plane matches by moving poses.

**Verdict.**
- Structural-plane gating (tall, wide, outermost planes only): built and measured. **No gain**
  (`out/diag/drift_structural.txt`): neutral on 42444949/50, worse p90 on 42444946 (3.9 -> 6.5 cm);
  injected drift still not removed. Kept as an option, off by default.
- Stronger robust loss (Cauchy): no measurable change. Off by default.
- Found a design flaw while reading the results against the paper: our odometry prior penalised each
  fragment's *absolute* correction, which caps any correction near 3 cm, so a 15 cm drift can never be
  removed. The pose-graph formulation trusts odometry *between consecutive fragments*. Implemented as
  `prior_mode="relative"`; results in `out/diag/drift_relative.txt`:

  | capture | injected drift, median / max cm: none -> relative 1 cm | real poses vs laser, median / p90 cm: ARKit -> relative 1 cm |
  |---|---|---|
  | 42444946 | 4.3 / 9.8 -> 4.5 / 9.2 | 2.4 / 3.9 -> 3.2 / 4.9 (harm) |
  | 42444949 | 3.9 / 16.6 -> 3.1 / 14.6 | 1.9 / 2.8 -> 2.0 / 3.3 |
  | 42444950 | 3.9 / 17.2 -> 2.4 / 14.0 | 2.2 / 4.7 -> 2.1 / 3.8 |

  Partial gain on injected drift for two captures, harm on 42444946's real poses: **not adopted as the
  default**. None of the variants corrects yaw, and the injected test includes 1.5 deg of yaw (up to
  ~8 cm across a room), so the remaining injected error is probably yaw; this still has to be separated
  (translation-only vs yaw-only injection).

## 2. Ceiling height accuracy and repeatability

**Literature.** Zea & Hanebeck (JAIF 2022/23, iPad Pro): depth always reads short, ~1-2 % of range;
errors are spatially correlated within a frame, so averaging pixels does not help, only independent
frames do. Repeatability of iOS LiDAR interior measurements: SD 0.6-1.4 cm (SAE 2021-01-0891). No
published ceiling-specific repeatability study.

**Checked against our data.** The measured ceilings of the three captures of room 421337 are 3.0594,
3.0649, 3.0627 m: **spread 0.55 cm (gate 1 cm)**. The earlier "1.2 cm" was the spread of the *errors*,
which also contains the laser reference's own variation (3.0578-3.0714 m), because each capture is
scored on the patch of ceiling under its own polygon.

**Verdict.** Report bias and spread separately (ISO 5725 style). The global depth-scale uncertainty
(se 0.4 %, about 1.2 cm on 3 m) is a *bias* term common to all captures; it cannot cause spread. Three
captures cannot establish a 1 cm spread with confidence; state the chi-square interval on the SD.

## 3. Interval calibration with few rooms

**Literature.** Normalised / locally weighted conformal (Lei et al.); hierarchical conformal for
grouped data (Dunn, Wasserman & Ramdas); Mondrian bins; training-conditional (PAC) margin (Vovk 2012);
Clopper-Pearson bounds on every coverage figure; interval (Winkler) score.

**Checked against our data.** The proposed main lever, a "fraction of the wall observed" difficulty
term, **does not predict error** on our 29 walls (Spearman rho = 0.01). The large errors (10-21 cm) are
mostly short, fully observed walls: wall length errors are endpoint errors, set by the neighbouring
walls' positions.

**Verdict.** Adopt room-level pooling, PAC margin, Clopper-Pearson reporting and an observed/inferred
split. Look for a difficulty feature tied to the neighbouring walls instead of the wall itself.

## 4. Video and photo metric scale

**Literature.** MapAnything predicts one metric scale per scene; its own paper reports ~16 % metric
scale error from images alone, ~2-5 % only with metric depth or poses as inputs. Its intrinsics input
conditions the model only weakly (open GitHub issues #38, #67, #93, #150). Best documented RGB-only
route: SfM poses on dense frames (150-300) + mono-metric depth median ratio (Depth Anything 3 Metric,
Apache) for scale, roughly 3-6 %; a physical reference (door leaf height 2.03 m, a known length) is the
only route found to 1-3 %.

**Verdict.** Adopt SfM on dense frames + mono-metric scale + door prior; report honestly if video
misses +-3 %. Check MPS vs CPU outputs first (a macOS 26 MPS attention regression is reported for M1).
Photo room area 1.6 vs 15 m2 is a 3x linear error: a layout bug, not a scale problem.

## 5. Photo multi-room stitching

**Literature.** Extreme SfM (Shabani ICCV 2021): door-based enumeration, 47 % top-1 / 78 % top-5. SALVe
(ECCV 2022): 57 % of panoramas localised. No production app stitches untracked photos automatically.

**Verdict.** Add a "bridge photo" per door to the protocol (one photo from inside room A showing the
door and room B, placed in both folders): it gives the A-to-B transform and relative scale directly.
Keep door matching as fallback, with exhaustive search, a free wall-gap variable (8-35 cm), joint
per-room scale and abstention. Report net vs gross area explicitly (walls are 8-12 % of gross).

## 6. Opening widths (<= 2 cm on 85 %)

**Literature.** Best TLS occupancy-gap method (Adan & Huber 2011): 36 % of edges within 2.5 cm. iPhone
LiDAR reads door frames ~1.4 cm narrow (FIG 2022). No published phone method shows <= 2 cm widths.

**Verdict.** Depth edge-bias correction and free-vs-occluded ray casting first (cheap, removes phantoms
and the systematic bias); multi-view high-res RGB jamb localisation next. Ground truth: ScanNet++ (Faro
laser + iPhone stream) needs an academic application by the user; until then openings carry
"conformal-transfer" intervals.

## 7. Video SfM stack on an M1 Mac (follow-up to section 4)

**Findings (verified in this venv unless marked).**
- pip `pycolmap` 4.2.1 is built without ONNX: creating an ALIKED extractor or the SIFT-LightGlue matcher
  raises an uncaught C++ error and aborts Python. Use SIFT only (conda-forge colmap has ONNX).
- `pycolmap` and `torch` each ship their own `libomp`; loading both in one process aborts ("OMP: Error
  #15"), in either import order (hloc issue #491). Run COLMAP and torch stages in separate processes.
  `KMP_DUPLICATE_LIB_OK=TRUE` is documented as unsafe and was not adopted.
- Fix intrinsics through `ImageReaderOptions(camera_model="PINHOLE", camera_params=...)` with
  `CameraMode.SINGLE`, and `ba_refine_focal_length/principal_point/extra_params = False`; sequential
  matching with overlap 15 on 150-300 dense frames. The earlier 3/32 registration used 32 sparse
  keyframes and per-image cameras.
- Metric depth for the scale step: Depth Pro (`infer(img, f_px=...)`, any torch device; Apple sample
  code licence) is the lowest-risk choice on a Mac. DA3METRIC-LARGE (Apache-2.0) runs on MPS only via
  a community fork (unverified here). MoGe-2's main branch does not install on macOS; UniDepthV2 is
  CUDA-only and CC BY-NC.
- MPS attention correctness: torch 2.14.1 on macOS 26.1 / M1 Pro matches the CPU reference (max error
  4.8e-7 fp32) on the upstream repro of pytorch#163597. The reported macOS 26 regression is not
  present on this setup.

**Incident.** While verifying this, the research agent ran pycolmap probes that triggered both aborts
above, producing several "Python quit unexpectedly" dialogs (crash reports 19:31-19:35). No project
files or packages were changed.

## 8. Photo tier: ceiling height and metric scale (not yet implemented)

**Measured first (6 ARKitScenes rooms, stand-in photos).** Wall lengths +4 %, +1 %, -3 % (gate +-8 %);
ceiling -11 %, 0 %, -59 % (a low horizontal surface taken as ceiling) and one 2.40 m fallback (no
ceiling seen: the stand-in photos exclude up-pitched views). Some laser references were themselves
wrong (e.g. ceiling 1.64 m): the scorer's registration is rigid, so a photo cloud with a scale error
mis-registers.

**Recommendations (research, unverified on our data).**
1. Ceiling from image junctions, scale-free: segment wall/floor/ceiling (ADE20K SegFormer), and per
   wall column use the angles to the ceiling line (beta) and floor line (alpha) from the gravity
   horizon: `H = h_cam * (1 + tan(beta) / tan(alpha))` (the camera-height ratio used by LayoutNet /
   DuLa-Net). Count 3D points as ceiling only where segmentation says ceiling.
2. Scale from door leaf height (US 2.032 m, UK 1.981, DE 1.985): about +-2-3 %; not door widths or
   switch/outlet heights (+-10 % or worse).
3. Camera-height prior: generic about +-8 %; with the user's stature about +-3 %. Calibrate against
   ARKit camera heights first.
4. Fuse MapAnything scale, door height, camera height and mono-metric depth (MoGe-2 Rel^p 4.4-5.6 %
   on NYUv2 / iBims) by inverse variance; aim +-3-5 %.
5. Standard ceiling heights only as a labelled prior with a wide interval (2.44 m [2.29, 2.74]).
6. Protocol: one level "height shot" per room in portrait or 0.5x so both junctions are in frame;
   one photo per door showing the whole leaf.

**Evaluation fix.** Score raw metric numbers first; estimate scale separately with a gravity-aligned,
4-yaw, scale-aware registration (Open3D point-to-point ICP `with_scaling=True`, initialised from room
corners, scale sanity range 0.7-1.3), and report scale error, shape error after scale, and raw error.
