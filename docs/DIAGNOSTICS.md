# LiDAR tier diagnostics (2026-10-04)

The question was why the LiDAR tier failed its gates on ARKitScenes 42444946 against the Faro laser:
- ceiling −3.0 cm (gate 1.5 cm);
- walls off by up to 17 cm;
- drift correction making the walls worse.

The order was research first, then one cheap experiment per hypothesis. Every number below can be
regenerated with the command in its row. Outputs go to `out/diag/` and are not committed.

## Hypotheses from the literature

| # | Hypothesis | Source |
|---|---|---|
| H1 | iPad/iPhone LiDAR reads short by 1–2% of range | Modeling Spatial Uncertainty for the iPad Pro Depth Sensor, JAIF 2024 |
| H2 | Drift correction absorbs uncorrected depth bias into the poses | CLAMS, Teichman, Miller & Thrun, RSS 2013 |
| H3 | ARKit trajectory scale or drift | ADVIO, ECCV 2018; four-VIO benchmark comparison, 2022 |
| H4 | The reference itself is uncertain, or matched the wrong surface | ARKitScenes, NeurIPS 2021 (reports 1–2 cm registration) |

## Results

### D1/D6: phone depth vs laser-rendered depth, per pixel, no poses
Command: `python -m roomscope.eval.depth_bias <scene>`

| Capture | Median relative error (confidence 2) | Per-frame IQR |
|---|---|---|
| 42444946 | −1.31% | −1.63 … −1.04% |
| 42444949 | −1.10% | −1.60 … −0.51% |
| 42444950 | −1.23% | −1.55 … −0.93% |

- **No dependence on incidence angle**, so the grazing-angle hypothesis is rejected.
- **Image radius:** −1.0% at the centre, −1.5% at the edges. That is a small intrinsics or distortion component.
- **Surface type:** walls −1.76%, floor and ceiling −1.08%.
- **Model:** the split into scale plus offset is unstable across captures, so the shipped model is a single scale.
- **Verdict:** H1 confirmed.

### Depth correction applied (scale 0.9883)
The scale is fitted on 42444949 and 42444950 only; 42444946 is the scored scene.

| 42444946, drift on | Ceiling error vs Faro |
|---|---|
| Before | −3.04 cm |
| After | +0.24 cm ✅ |

### D4: the two walls off by 15–17 cm
Command: `python bench/diag_lidar.py <scene> --affine A B`

- **W9:** the wall was placed on a wardrobe face, whose laser surface stops at 1.97 m. The real full-height wall is 15.7 cm behind it.
- **W10:** the wall was placed on a curtain: a sparse, full-height surface, densest at the rail (2.75–3.0 m). The real wall, which has a window, is 17.3 cm behind it.
- **Verdict:** a pipeline classification error (wall vs furniture or curtain), not depth or drift. The reference chose correctly.

### D3: ARKit pose error per frame
Laser-rendered frames are placed with ARKit poses and aligned onto the Faro cloud by ICP.
Command: `python -m roomscope.eval.pose_drift <scene> <nodrift_out>`

- **Camera position error:** median 2.6 cm, 90th percentile 4.7 cm, max 9.7 cm, slow and structured over time.
- **Rotation error:** median 0.7°.
- **Trajectory scale:** −0.42%.
- **Verdict:** H3 is real but small. ARKit's own poses are already good to about 2.5 cm.

### D5: drift-correction variants scored on camera pose vs laser
Command: `python bench/drift_eval.py <scene> <nodrift_out>`

On 42444946:

| Variant | Median | p90 | Rotation |
|---|---|---|---|
| ARKit as-is | 2.4 cm | 3.8 cm | 0.72° |
| Current (per-fragment yaw, 20 cm match) | 8.2 cm | 38.9 cm | 1.62° |
| No per-fragment yaw | 4.4 cm | 12.4 cm | 0.72° |
| No yaw, 8 cm match | 4.0 cm | 8.6 cm | 0.72° |

Verdict:
- Our drift correction degrades good odometry. The per-fragment yaw re-estimate is the worst part.
- The 20 cm plane-matching window lets furniture faces pull fragments.
- H2 is not the explanation: correcting depth did not rescue the drift-on result.

### In-scan self-calibration of the depth scale (negative result)
Method: CLAMS-style plane consistency across camera distance (`core/depth_calib.py`, unused).

- **Estimates:** +3.07%, +0.19% and +0.20%, against laser truth −1.31%, −1.10% and −1.23%.
- **Fixes tried:**
  - Short-window differencing: estimates ranged −2.0% to +3.9%.
  - A joint fit of pose-to-depth latency: did not help.
- **Why it fails:** the signal is too small for these captures.
  - The camera-to-plane distance varies by only 0.2–0.4 m, so a 1.2% scale error moves a plane by 2–5 mm.
  - ARKit pose drift is 2–3 cm.
- **Consequence:** the shipped correction is a per-device prior plus an optional tape override (`--depth-scale`).

### Ceiling with drift off: a phantom layer
- The laser ceiling is a single flat level at 3.06 m.
- Our drift-off cloud has the true layer at 3.06–3.08 m and a phantom at 3.14–3.16 m holding 15% of the points. The phantom comes from frames with an ARKit height error.
- The level picker took the highest layer. It now takes the best-supported layer within 15 cm of the extreme.
