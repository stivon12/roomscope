"""Per-device LiDAR depth scale: config lookup, interval widening for uncalibrated devices, and the
calibrate-depth reference fit."""
from __future__ import annotations

import shutil

import pytest

from roomscope.core import calibrate as Cal
from roomscope.core import depth_calib as DC


def test_calibration_device_adds_no_excess():
    ds = DC.resolve_depth_scale("iPad Pro (2020)")
    assert ds.device == "iPad Pro (2020)" and ds.excess_se == 0.0


def test_unknown_device_gets_wider_prior():
    ds = DC.resolve_depth_scale(None)
    assert ds.device == "unknown" and ds.excess_se > 0.005
    assert DC.resolve_depth_scale("iPhone 99 Pro").device == "unknown"


def _meas(v, half, **kw):
    return {"value": v, "lo": v - half, "hi": v + half, "unit": "m", "level": 0.9, **kw}


def _result():
    room = {"id": "R1", "ceiling_height": _meas(3.0, 0.02), "walls": [{"id": "R1-W1", "length": _meas(4.0, 0.02)}],
            "floor_area": _meas(12.0, 0.1), "openings": []}
    return {"rooms": [room], "surfaces": [], "footprint": {"area": _meas(12.0, 0.1)}}


def test_widening_is_proportional_and_skips_priors():
    r = Cal.widen_for_depth_scale(_result(), 0.01)
    c = r["rooms"][0]["ceiling_height"]
    assert c["hi"] - c["value"] == pytest.approx((0.02 ** 2 + (1.645 * 0.03) ** 2) ** 0.5, abs=2e-4)
    fa = r["rooms"][0]["floor_area"]
    assert fa["hi"] - fa["value"] > 1.645 * 0.02 * 12.0 * 0.99      # area: twice the relative error
    prior = _result()
    prior["rooms"][0]["ceiling_height"] = _meas(2.8, 0.8, observed=False)
    assert Cal.widen_for_depth_scale(prior, 0.01)["rooms"][0]["ceiling_height"]["hi"] == pytest.approx(3.6)


def test_reference_fit_recovers_scale():
    r = _result()
    r["rooms"][0]["ceiling_height"] = _meas(3.0 * 0.988, 0.008)
    a, se, _ = DC.scale_from_references(r, {"R1:ceiling": 3.0})
    assert a == pytest.approx(0.988, abs=1e-6) and 0.001 <= se < 0.005
    r["rooms"][0]["ceiling_height"]["observed"] = False
    with pytest.raises(ValueError):
        DC.scale_from_references(r, {"ceiling": 3.0})


def test_write_device_entry_keeps_comments(tmp_path):
    p = tmp_path / "depth_scale.yaml"
    shutil.copy(DC.CONFIG, p)
    DC.write_device_entry("iPhone 15 Pro", 0.9871, 0.0015, "test", p)
    DC.write_device_entry("iPhone 15 Pro", 0.9875, 0.0012, "test again", p)
    import yaml
    cfg = yaml.safe_load(p.read_text())
    assert cfg["devices"]["iPhone 15 Pro"] == {"scale": 0.9875, "se": 0.0012, "source": "test again"}
    assert "iPad Pro (2020)" in cfg["devices"] and "# Interval rule" in p.read_text()
