import json

import numpy as np
import pandas as pd
import pytest

from microgrid.calibration import IntervalCalibrator, by_source_horizon
from microgrid.data import synthetic
from microgrid.feasibility import perfect_foresight_episode
from microgrid.india import parse_power, prepare_india
from microgrid.physics import BatteryConfig


def weather_payload():
    keys = pd.date_range("2020-01-01", periods=24, freq="h").strftime("%Y%m%d%H")
    return {
        "header": {"time_standard": "UTC", "fill_value": -999},
        "parameters": {
            "ALLSKY_SFC_SW_DWN": {"units": "Wh/m^2"},
            "T2M": {"units": "C"},
            "WS10M": {"units": "m/s"},
        },
        "properties": {
            "parameter": {
                name: dict.fromkeys(keys, value)
                for name, value in [("ALLSKY_SFC_SW_DWN", 300), ("T2M", 25), ("WS10M", 7)]
            }
        },
    }


def test_power_units_missing_and_time_standard():
    p = weather_payload()
    f = parse_power(p)
    assert len(f) == 24 and f.irradiance_wm2.eq(300).all()
    p["header"]["time_standard"] = "LST"
    with pytest.raises(ValueError, match="UTC"):
        parse_power(p)
    p = weather_payload()
    p["parameters"]["ALLSKY_SFC_SW_DWN"]["units"] = "kWh/m^2/day"
    with pytest.raises(ValueError, match="units"):
        parse_power(p)
    p = weather_payload()
    p["properties"]["parameter"]["T2M"]["2020010100"] = -999
    with pytest.raises(ValueError, match="Missing"):
        parse_power(p)


def test_interval_correction_fit_boundary_and_heldout_coverage():
    truth = np.full((20, 24, 3), 10.0)
    q = np.zeros((20, 24, 3, 3))
    q[..., 0], q[..., 1], q[..., 2] = 2, 5, 8
    c = IntervalCalibrator().fit(truth, q, fit_end=480)
    assert (c.correction == 2).all()
    with pytest.raises(ValueError, match="overlaps"):
        c.apply(q[0], 479)
    corrected = np.array([c.apply(x, 480) for x in q])
    assert by_source_horizon(truth, corrected)["pooled"]["coverage_80"] == 1
    assert by_source_horizon(truth, q)["pooled"]["coverage_80"] == 0
    assert np.array_equal(corrected[..., 1], q[..., 1])
    with pytest.raises(ValueError, match="Too few"):
        IntervalCalibrator(alpha=0.01).fit(truth, q, 480)


def test_adequacy_lp_known_shortage_and_terminal_sensitivity():
    f = synthetic(72)
    f[["pv_kw", "wind_kw"]] = 0
    f["load_kw"] = 2
    trace, r = perfect_foresight_episode(f)
    assert r["eens_kwh"] == pytest.approx(144 - 40 * 0.95)
    assert trace.energy_kwh.iloc[-1] == pytest.approx(10)
    _, hold = perfect_foresight_episode(f, terminal="initial")
    assert hold["eens_kwh"] == pytest.approx(144)
    assert r["max_balance_residual_kw"] < 1e-7
    _, limited = perfect_foresight_episode(f, BatteryConfig(power_kw=1))
    assert limited["minimum_power_shortage_bound_kwh"] == 72
    assert limited["eens_kwh"] >= 72


def test_india_preparation_train_only_sizing_and_strict_join(tmp_path):
    f = synthetic(720)
    weather = tmp_path / "weather.parquet"
    f.drop(columns=["pv_kw", "wind_kw", "load_kw"]).to_parquet(weather, index=False)
    load = tmp_path / "load.csv"
    f[["timestamp", "load_kw"]].to_csv(load, index=False)
    meta = tmp_path / "meta.json"
    metadata = {
        "source_url": "https://example.invalid/test-fixture",
        "licence": "test fixture only",
        "site": "synthetic fixture, not India observations",
        "timezone": "UTC",
        "interval_convention": "start",
        "load_kind": "measured_local",
        "wind_height_assumption": "10m engineering fixture",
        "weather_terms_reference": "fixture",
        "units": "kW",
        "reuse_permitted": True,
    }
    meta.write_text(json.dumps(metadata))
    a = prepare_india(weather, load, meta, tmp_path / "a")
    changed = f[["timestamp", "load_kw"]].copy()
    changed.loc[500:, "load_kw"] *= 10
    changed.to_csv(load, index=False)
    b = prepare_india(weather, load, meta, tmp_path / "b")
    assert np.array_equal(a.pv_kw, b.pv_kw)
    assert np.array_equal(a.wind_kw, b.wind_kw)
    changed["timestamp"] += pd.Timedelta(minutes=30)
    changed.to_csv(load, index=False)
    with pytest.raises(ValueError, match="align"):
        prepare_india(weather, load, meta, tmp_path / "c")
    metadata["reuse_permitted"] = False
    meta.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="permission"):
        prepare_india(weather, load, meta, tmp_path / "c")
