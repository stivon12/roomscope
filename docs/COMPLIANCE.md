# Compliance matrix

Status: ✅ done & verified · 🟡 partial · ⬜ not started · ❌ known fail (see report)

| # | Requirement (brief §) | File path | Artifact | Status |
|---|---|---|---|---|
| 1.1 | Capture route: stock protocol (§1 Route 2) | `docs/PROTOCOL.md` | one-page protocol | 🟡 |
| 1.2 | Photo tier: 2–8 stills/room, per-room folders → stitched plan | `src/roomscope/frontends/photo.py`, `src/roomscope/core/stitch.py` | `out/<cap>/result.json` | 🟡 runs end to end (MapAnything + known-K rays + Manhattan view snap + single-room hull). 42444946: area 19.5 vs 17.6 m² (+11%; −4% vs the room's convex hull), ceiling −9.4%. Multi-capture evaluation running. Stitch untested on a real multi-room capture |
| 1.3 | Video tier: handheld walkthrough | `src/roomscope/frontends/video.py`, `sfm_worker.py` | `out/<cap>/result.json` | 🟡 runs end to end (MapAnything on 32 keyframes, default). COLMAP SfM path opt-in: degenerate on low-texture rooms (docs/RESEARCH.md §4, §7). Accuracy not yet within ±3% |
| 1.4 | LiDAR tier: depth+poses+intrinsics | `src/roomscope/frontends/lidar.py` (Stray + ARKitScenes loaders) | `out/<cap>/result.json` | 🟡 runs on all 9 ARKitScenes captures (7 rooms) scored vs Faro laser; Stray loader untested on a real export |
| 1.5 | Device matrix (tier × hardware × honest accuracy) | `docs/DEVICE_MATRIX.md` | table | ⬜ |
| 2.1 | Per-room walls, ceiling height, floor area, openings | `src/roomscope/core/layout.py` | result.json `rooms[]` | 🟡 LiDAR tier; unverified vs laser |
| 2.2 | Stitched multi-room plan, correct adjacency | `src/roomscope/core/stitch.py` | result.json `footprint`, `adjacency` | ⬜ |
| 2.3 | Per-surface damage regions, class + metric extent | `src/roomscope/damage/` | result.json `damage_regions[]` | ⬜ |
| 2.4 | Concealed-damage flags with rule fired | `rules/concealed.yaml` | result.json `concealed_flags[]` | ⬜ |
| 2.5 | Scope line items keyed to surfaces | `rules/scope.yaml` | result.json `scope_items[]` | ⬜ |
| 2.6 | Confidence interval on every measurement | `src/roomscope/core/calibrate.py`, `config/calibration.json` | schema `$defs/measurement` | 🟡 LiDAR: room-weighted split conformal, Mondrian wall bins by neighbour support, PAC margin; LORO coverage 0.94/0.95 (walls), 1.00 [0.48–1.00] (ceiling, ±2.2 cm). Photo/video: fits pending |
| 2.7 | One command per capture | `src/roomscope/cli.py` | `roomscope run <capture>` | 🟡 |
| 2.8 | JSON to published schema | `schema/plan.schema.json` | `roomscope validate` | 🟡 |
| 2.9 | Rendered plan | `src/roomscope/render.py` | `out/<cap>/plan.png` | ✅ |
| B.1 | Multi-room capture ≥3 rooms + connector | `benchmark/raw/MANIFEST.md` | raw data | ⬜ |
| B.2 | Furnished room, staged damage, 2 classes | `benchmark/raw/MANIFEST.md` | raw data | ⬜ |
| B.3 | Same rooms at all 3 tiers (photo as per-room folders) | `benchmark/raw/MANIFEST.md` | raw data | ⬜ |
| B.4 | ≥1 room captured twice per tier | `benchmark/raw/MANIFEST.md` | raw data | ⬜ |
| B.5 | Laser/tape ground truth on everything | `benchmark/gt/*.yaml` | GT files | ⬜ |
| G.1 | Openings ≤2 cm on ≥85%, misses+phantoms scored | `bench/gates.py` | `bench/results/gates.md` | ⬜ |
| G.2 | Ceiling ≤1.5 cm; multi-capture spread ≤1 cm; bias vs variance stated | `config/depth_scale.yaml`, `src/roomscope/eval/depth_bias.py`, `fix/` | test vs Faro laser | 🟡 accuracy: +0.15 / +0.55 / −0.76 cm on 42444946/49/50, each with a depth scale fitted on the other two (fix 1). Spread of the measured ceilings 0.55 cm (gate 1 cm); 3 captures cannot establish it with confidence. Bias term (scale se 0.4% ≈ 1.2 cm) reported separately |
| G.3 | Repeatability 1 cm or 0.5%/wall | `bench/repeatability.py` | `bench/results/repeatability.md` | ⬜ |
| G.4 | Drift handling + on/off ablation | `src/roomscope/core/drift.py`, `bench/drift_eval.py`, `bench/drift_inject.py` | result.json `meta.drift_correction` | 🟡 default is do-no-harm (real poses vs laser: 2.4→2.5, 1.9→1.9, 2.2→1.9 cm). Large injected drift not removed (max 16.6→15.5 cm). Tried and not adopted: structural-plane gating, Cauchy loss, relative odometry prior (docs/RESEARCH.md §1, out/diag/drift_*.txt) ❌ |
| G.5 | Photo-tier whole-property stitch, ±8% footprint, calibrated | `bench/gates.py` | `bench/results/gates.md` | ⬜ |
| G.6 | Photo ±8% / video ±3% walls; calibration at every tier | `bench/calibration.py` | `bench/results/calibration.md` | ⬜ |
| 3.1 | Head-to-head vs Polycam on 2 rooms, ≥70% beat/tie | `bench/h2h.py` | `bench/results/h2h.md` + `benchmark/raw/polycam/` | ⬜ |
| 4.1 | Fix declaration (worst gate, root cause, prediction) | `fix/DECLARATION.md` | doc | ✅ fix 1 (depth bias; retrospective, disclosed) and fix 2 (wall-length intervals; declared and tagged `fix2-before` before code) |
| 4.2 | Fix shipped; regenerable before/after + diff | `fix/`, `bench/fix_loop.py`, tags `fix2-before`/`fix2-after` | runs + diff | ✅ fix 1: ceiling prediction met 3/3, wall prediction met 1/3. Fix 2: 4 of 5 predicted numbers met, supported-bin width missed by 0.7 cm |
| 5.1 | Incremental commit history | git log | — | ✅ one commit per verified change, each with measured outcome |
| D.3 | README: fresh capture → result <15 min, clean machine | `README.md` | — | ⬜ |
| D.4 | Reproduction bundle (deterministic cache + live path) | `bench/run_all.py`, `cache/` | — | ⬜ |
| D.7 | Technical report ≤6 pages | `docs/REPORT.md` | PDF | ⬜ |
| C.1 | Mirrors, glass, wet-look, low light covered | `src/roomscope/core/filters.py`, report §failure modes | — | ⬜ |
| C.2 | Weights fetched by script; models disclosed | `scripts/fetch_weights.sh`, result.json `meta.models` | — | 🟡 MapAnything (Apache) fetched by script and disclosed; VGGT tested, not adopted (non-commercial, marginal gain) |
