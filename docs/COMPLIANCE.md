# Compliance matrix

Status: ✅ done & verified · 🟡 partial · ⬜ not started · ❌ known fail (see report)

| # | Requirement (brief §) | File path | Artifact | Status |
|---|---|---|---|---|
| 1.1 | Capture route: stock protocol (§1 Route 2) | `docs/PROTOCOL.md` | one-page protocol | 🟡 |
| 1.2 | Photo tier: 2–8 stills/room, per-room folders → stitched plan | `src/roomscope/frontends/photo.py` | `out/<cap>/result.json` | ⬜ |
| 1.3 | Video tier: handheld walkthrough | `src/roomscope/frontends/video.py` | `out/<cap>/result.json` | ⬜ |
| 1.4 | LiDAR tier: depth+poses+intrinsics | `src/roomscope/frontends/lidar.py` (Stray + ARKitScenes loaders) | `out/<cap>/result.json` | 🟡 runs on real ARKitScenes 41069042; laser scoring pending GT scene |
| 1.5 | Device matrix (tier × hardware × honest accuracy) | `docs/DEVICE_MATRIX.md` | table | ⬜ |
| 2.1 | Per-room walls, ceiling height, floor area, openings | `src/roomscope/core/layout.py` | result.json `rooms[]` | 🟡 LiDAR tier; unverified vs laser |
| 2.2 | Stitched multi-room plan, correct adjacency | `src/roomscope/core/stitch.py` | result.json `footprint`, `adjacency` | ⬜ |
| 2.3 | Per-surface damage regions, class + metric extent | `src/roomscope/damage/` | result.json `damage_regions[]` | ⬜ |
| 2.4 | Concealed-damage flags with rule fired | `rules/concealed.yaml` | result.json `concealed_flags[]` | ⬜ |
| 2.5 | Scope line items keyed to surfaces | `rules/scope.yaml` | result.json `scope_items[]` | ⬜ |
| 2.6 | Confidence interval on every measurement | `src/roomscope/measure.py`, `src/roomscope/core/calibrate.py` | schema `$defs/measurement` | 🟡 |
| 2.7 | One command per capture | `src/roomscope/cli.py` | `roomscope run <capture>` | 🟡 |
| 2.8 | JSON to published schema | `schema/plan.schema.json` | `roomscope validate` | 🟡 |
| 2.9 | Rendered plan | `src/roomscope/render.py` | `out/<cap>/plan.png` | ✅ |
| B.1 | Multi-room capture ≥3 rooms + connector | `benchmark/raw/MANIFEST.md` | raw data | ⬜ |
| B.2 | Furnished room, staged damage, 2 classes | `benchmark/raw/MANIFEST.md` | raw data | ⬜ |
| B.3 | Same rooms at all 3 tiers (photo as per-room folders) | `benchmark/raw/MANIFEST.md` | raw data | ⬜ |
| B.4 | ≥1 room captured twice per tier | `benchmark/raw/MANIFEST.md` | raw data | ⬜ |
| B.5 | Laser/tape ground truth on everything | `benchmark/gt/*.yaml` | GT files | ⬜ |
| G.1 | Openings ≤2 cm on ≥85%, misses+phantoms scored | `bench/gates.py` | `bench/results/gates.md` | ⬜ |
| G.2 | Ceiling ≤1.5 cm; multi-capture spread ≤1 cm; bias vs variance stated | `src/roomscope/eval/laser.py`, `tests/test_lidar_real.py` | test vs Faro laser | 🟡 test written, awaiting GT scene 42444946 |
| G.3 | Repeatability 1 cm or 0.5%/wall | `bench/repeatability.py` | `bench/results/repeatability.md` | ⬜ |
| G.4 | Drift handling + on/off ablation | `src/roomscope/core/drift.py`, `--drift/--no-drift`, `tests/test_lidar_real.py` | result.json `meta.drift_correction` | 🟡 plane residual 6.4→3.0 cm on 41069042; laser ablation pending |
| G.5 | Photo-tier whole-property stitch, ±8% footprint, calibrated | `bench/gates.py` | `bench/results/gates.md` | ⬜ |
| G.6 | Photo ±8% / video ±3% walls; calibration at every tier | `bench/calibration.py` | `bench/results/calibration.md` | ⬜ |
| 3.1 | Head-to-head vs Polycam on 2 rooms, ≥70% beat/tie | `bench/h2h.py` | `bench/results/h2h.md` + `benchmark/raw/polycam/` | ⬜ |
| 4.1 | Fix declaration (worst gate, root cause, prediction) | `fix/DECLARATION.md` | doc | ⬜ |
| 4.2 | Fix shipped; regenerable before/after + diff | `fix/` + git tags `fix-before`/`fix-after` | runs + diff | ⬜ |
| 5.1 | Incremental commit history | git log | — | 🟡 |
| D.3 | README: fresh capture → result <15 min, clean machine | `README.md` | — | ⬜ |
| D.4 | Reproduction bundle (deterministic cache + live path) | `bench/run_all.py`, `cache/` | — | ⬜ |
| D.7 | Technical report ≤6 pages | `docs/REPORT.md` | PDF | ⬜ |
| C.1 | Mirrors, glass, wet-look, low light covered | `src/roomscope/core/filters.py`, report §failure modes | — | ⬜ |
| C.2 | Weights fetched by script; models disclosed | `scripts/fetch_weights.sh`, result.json `meta.models` | — | ⬜ |
