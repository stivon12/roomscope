# Compliance matrix

Status: ✅ done and verified · 🟡 partial · ⬜ not started · ❌ known fail (see the report)

| # | Requirement | File path | Artifact | Status |
|---|---|---|---|---|
| 1.1 | Capture route: stock protocol (Route 2) | `CAPTURE_PROTOCOL.md` | one-page protocol | ✅ Camera app (photo, video) + Stray Scanner (LiDAR); hand-off steps included |
| 1.2 | Photo tier: 2–8 stills per room, per-room folders, stitched plan | `src/roomscope/frontends/photo.py`, `src/roomscope/core/stitch.py` | `out/<cap>/result.json` | 🟡 runs on every benchmark capture (diagonal-outline crash fixed, 7ebd9cf); multi-room stitch does not connect rooms (G.5) |
| 1.3 | Video tier: handheld walkthrough | `src/roomscope/frontends/video.py`, `recon.py` | `out/<cap>/result.json` | 🟡 runs on single rooms; multi-room walk keeps one COLMAP piece ❌ |
| 1.4 | LiDAR tier | `src/roomscope/frontends/lidar.py` | `out/<cap>/result.json` | ✅ Stray (own iPhone Pro, MuSHRoom iPhone) and ARKitScenes |
| 1.5 | Device matrix | `docs/DEVICE_MATRIX.md` | table | ✅ |
| 2.1 | Per-room walls, ceiling height, floor area, openings | `src/roomscope/core/layout.py` | result.json `rooms[]` | 🟡 all produced; walls median 8.7 cm vs laser; openings unmeasured |
| 2.2 | Stitched multi-room plan, adjacency | `src/roomscope/core/stitch.py`, `pipeline.py` | `footprint`, `adjacency` | 🟡 LiDAR 6 rooms with adjacency on YC; photo 6 rooms, 1 connected; video 1 room |
| 2.3 | Per-surface damage regions, class and metric extent | `src/roomscope/damage/` | `damage_regions[]` | 🟡 LiDAR tier; real-photo check (`bench/damage_photos.py`); extent unmeasured (no damaged room) |
| 2.4 | Concealed-damage flags with the rule that fired | `rules/concealed.yaml`, `src/roomscope/scope/` | `concealed_flags[]` | ✅ public citations (EPA, BRE, 40 CFR 745); `tests/test_scope_rules.py` |
| 2.5 | Scope line items keyed to surfaces | `rules/scope.yaml`, `src/roomscope/scope/` | `scope_items[]` | ✅ category hints from the public Xactimate list, no prices |
| 2.6 | Confidence interval on every measurement | `src/roomscope/core/calibrate.py`, `config/calibration.json` | schema `$defs/measurement` | 🟡 every tier calibrated. Benchmark coverage: LiDAR 13/15, video 4/4 (±25 %), photo 3/3 (±103 %). LiDAR still fitted against the old wall reference |
| 2.7 | One command per capture | `src/roomscope/cli.py` | `roomscope run <capture>` | ✅ |
| 2.8 | JSON to the published schema | `schema/plan.schema.json` | `roomscope validate` | ✅ |
| 2.9 | Rendered plan | `src/roomscope/render.py` | `plan.png` | ✅ |
| B.1 | Multi-room capture, 3+ rooms and a connector | `benchmark/SET.md`, `bench/set.yaml` | YC c7d28f72c6 | 🟡 no ground truth |
| B.2 | Furnished room, staged damage, 2 classes | `benchmark/SET.md` | — | ⬜ needs a physical capture |
| B.3 | Same rooms at all 3 tiers (photo as per-room folders) | `bench/set.yaml`, `benchmark/photo_select/` | 5 captures × 3 tiers | 🟡 video and photo come from the same recordings as the LiDAR capture |
| B.4 | One room captured twice | `bench/set.yaml` | 421337 ×2, honka ×2 | ✅ |
| B.5 | Laser or tape ground truth on everything | `benchmark/SET.md` | Faro laser (4 captures) | 🟡 none for YC, openings or damage |
| G.1 | Openings ≤ 2 cm on ≥ 85 % | `bench/gates.py` | `bench/results/gates.md` | ⬜ not measurable: no opening ground truth |
| G.2 | Ceiling ≤ 1.5 cm; repeat spread ≤ 1 cm | `bench/gates.py` | `bench/results/gates.md` | ✅ 4/4 rooms; spread 0.62 / 0.41 cm (repeatable and unbiased at this n) |
| G.3 | Repeatability: 1 cm or 0.5 % per wall | `bench/gates.py`, `bench/repeatability.py` | `bench/results/gates.md` | ❌ LiDAR 4/12, video 0/4, photo 0/2 |
| G.4 | Drift handling + on/off ablation of the stitched footprint | `src/roomscope/core/drift.py`, `out/bench/drift_off` | `TECHNICAL_REPORT.md` §3 | 🟡 plane-anchored pose graph; do-no-harm on laser poses; YC footprint ablation in the report |
| G.5 | Photo whole-property stitch, ±8 % footprint, calibrated | `bench/gates.py` | `bench/results/gates.md` | ❌ 6 of 6 rooms reconstructed but only 1 connected: no doors detected at the photo tier; footprint not measurable (no ground truth) |
| G.6 | Photo ±8 % / video ±3 % walls; calibration at every tier | `bench/gates.py`, `bench/calibrate.py` | `bench/results/gates.md` | ❌ accuracy not met (scale off 3–12 %); calibration met at every tier after Fix 3 (wide) |
| 3.1 | Head-to-head vs a consumer app on 2 rooms | — | — | ⬜ needs Polycam captures of the benchmark rooms |
| 4.1 | Fix declaration (worst gate, root cause, prediction) | `fix/DECLARATION.md` | doc, tags `fix2-before`, `fix3-before` | ✅ fixes 1–3 |
| 4.2 | Fix shipped; regenerable before/after + diff | `fix/`, `bench/gates.py`, `bench/calibrate.py` | tags `fix3-before` / `fix3-after` | ✅ Fix 3: coverage 0/7 → 7/7 (tags `fix3-before` / `fix3-after`); widths missed the prediction, explained |
| 5.1 | Incremental commit history | git log | — | ✅ |
| D.3 | README: fresh capture → result in under 15 min | `README.md` | — | 🟡 measured 4 min 20 s before the damage models (+5 GB, not re-timed) |
| D.4 | Reproduction bundle | `bench/gates.py`, caches under `out/cache` | — | 🟡 regeneration commands at the top of `TECHNICAL_REPORT.md`; raw data packaging not done |
| D.7 | Technical report ≤ 6 pages | `TECHNICAL_REPORT.md`, `scripts/render_report.py` | `TECHNICAL_REPORT.pdf` (4 pages) | ✅ |
| D.8 | Raw benchmark data | `benchmark/raw/MANIFEST.md` | — | 🟡 listed and sourced; own captures pending |
| C.1 | Mirrors, glass, wet-look, low light covered | `TECHNICAL_REPORT.md` §7, `frontends/lidar.py` | — | 🟡 covered in writing; glass and low-light geometry not mitigated |
| C.2 | Weights fetched by script; models disclosed | `scripts/fetch_weights.sh`, `meta.models` | — | ✅ all ungated, Apache/MIT |
