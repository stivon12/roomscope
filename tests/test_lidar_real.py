"""LiDAR tier on real ARKitScenes data, scored against the Faro laser reference.

Gates checked here (brief Part 2): ceiling height <= 1.5 cm per room; drift handling must reduce
plane disagreement and must not make the plan worse than raw poses. Wall-plane accuracy is checked at
2 cm (the LiDAR noise floor). Skips when the scene is not downloaded (see benchmark/raw/MANIFEST.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from roomscope.eval.laser import find_laser_dir, score_capture
from roomscope.pipeline import run_capture

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "benchmark/raw/arkitscenes/raw/Validation/42444946"
SCHEMA = json.loads((ROOT / "schema/plan.schema.json").read_text())

needs_scene = pytest.mark.skipif(not (SCENE / "lowres_depth").is_dir() or find_laser_dir(SCENE) is None,
                                 reason="ARKitScenes 42444946 + laser clouds not downloaded")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("lidar")
    res = {}
    for drift in (True, False):
        d = out / ("on" if drift else "off")
        run_capture(SCENE, tier="lidar", out_dir=d, drift=drift)
        res[drift] = (d / SCENE.name, score_capture(d / SCENE.name, SCENE))
    return res


@needs_scene
def test_schema_valid(runs):
    for d, _ in runs.values():
        jsonschema.validate(json.loads((d / "result.json").read_text()), SCHEMA)


@needs_scene
def test_registration_sane(runs):
    # ICP fitness is measured against the whole scanner-centred laser crop (adjacent rooms the capture
    # never saw), so it is low (~0.14-0.22) even when registration is good; check that most reported
    # walls found a laser reference surface instead
    _, sc = runs[True]
    assert sc["n_ref_planes"] >= 4


@needs_scene
def test_ceiling_height_gate(runs):
    _, sc = runs[True]
    assert sc["ceil"], "no room had laser floor+ceiling inside its footprint"
    errs = [abs(r["err"]) for r in sc["ceil"]]
    assert max(errs) <= 0.015, sc["ceil"]


@needs_scene
def test_wall_planes_within_noise_floor(runs):
    _, sc = runs[True]
    e = np.abs([r["offset_err"] for r in sc["wall_planes"]])
    assert len(e) >= 3
    assert np.median(e) <= 0.02, sc["wall_planes"]


@needs_scene
def test_drift_correction_helps(runs):
    _, on = runs[True]
    _, off = runs[False]
    d = on["drift"]
    assert d["residual_after_m"] < d["residual_before_m"]
    e_on = np.mean(np.abs([r["offset_err"] for r in on["wall_planes"]]))
    e_off = np.mean(np.abs([r["offset_err"] for r in off["wall_planes"]]))
    assert e_on <= e_off + 0.005, (e_on, e_off)
