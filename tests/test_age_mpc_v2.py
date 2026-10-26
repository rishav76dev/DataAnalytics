import numpy as np
import pandas as pd
import pytest

from microgrid.age_mpc import age_mpc, wear_coefficients
from microgrid.ageing import effective_age_increment
from microgrid.data import synthetic
from microgrid.phase_forecasts import daylight, feature_matrix
from microgrid.physics import BatteryConfig
from microgrid.trajectory_calibration import OnlineDeficitEnvelope, intervals


def test_feature_matrix_does_not_read_future_power():
    frame = synthetic(720)
    x, base, _ = feature_matrix(frame, [300])
    frame.loc[300:, ["pv_kw", "wind_kw", "load_kw"]] = 9999
    y, other, _ = feature_matrix(frame, [300])
    assert np.array_equal(x, y) and np.array_equal(base, other)


def test_solar_mask_respects_local_midnight_and_noon():
    times = pd.DatetimeIndex(["2024-06-01T00:00:00+05:30", "2024-06-01T12:00:00+05:30"])
    assert daylight(times).tolist() == [False, True]


def test_calendar_surrogate_upper_bounds_reference_exposure():
    from microgrid.ageing import graphite_potential

    c = BatteryConfig(capacity_kwh=400)
    coeff = wear_coefficients(c)
    assert coeff["calendar_per_kwh_hour"] > 0 and coeff["cyclic_per_dc_kwh"] > 0
    assert coeff["upper_envelope_gap_max"] >= 0
    soc = np.linspace(c.soc_min, c.soc_max, 161)
    k = 8.37e4 * np.exp(-5210 / 298.15) * np.exp(-3560 * graphite_potential(soc) / 298.15)
    actual = k * ((100 + 1 / 24) ** 0.526 - 100**0.526)
    proxy = coeff["calendar_per_kwh_hour"] * c.capacity * soc
    proxy += coeff["calendar_intercept_per_hour"]
    assert np.all(proxy >= actual - 1e-15)


def test_mpc_immediate_service_is_hard_and_common_stage_rejected():
    s = np.array([[[0, 0, 12], [30, 0, 0], [0, 0, 20]]], float)
    assert age_mpc(s, 50) == pytest.approx(12)
    s = np.repeat(s, 2, axis=0)
    s[1, 0, 2] = 10
    with pytest.raises(ValueError, match="First-stage"):
        age_mpc(s, 50)


def test_mpc_charges_surplus_with_physical_bounds_and_reserve():
    s = np.array([[20, 0, 0], [0, 0, 20], [0, 0, 20]], float)
    a = age_mpc(s, 30, reserve=np.array([70, 40, 10], float))
    assert -20 - 1e-7 <= a <= 0


def test_effective_age_integration_is_time_step_invariant():
    k, p = 0.0005, 0.526
    loss = 0.0
    for _ in range(240):
        loss += effective_age_increment(loss, 0.1, k, p)
    assert loss == pytest.approx(k * 24**p, rel=1e-12)


def test_online_calibration_delays_every_future_label():
    frame = synthetic(120)
    s = np.repeat(frame[["pv_kw", "wind_kw", "load_kw"]].iloc[50:74].to_numpy()[None], 100, axis=0)
    cal = OnlineDeficitEnvelope(min_days=20)
    cal.predict(50, frame.timestamp.iloc[50], s, 100, 10, 90)
    cal.advance(73, frame)
    assert len(cal.pending) == 1 and sum(map(len, cal.scores)) == 0
    cal.advance(74, frame)
    assert not cal.pending and sum(map(len, cal.scores)) == 1
    assert sum(sum(abs(x) for x in q) for q in cal.scores) == pytest.approx(0)


def test_interval_envelope_includes_every_contiguous_interval():
    x = np.array([1.0, -2.0, 4.0])
    value, i, j = intervals(x)
    assert len(value) == 6
    assert value[(i == 0) & (j == 2)][0] == 3
    assert value[(i == 2) & (j == 2)][0] == 4


def test_scenario_age_mpc_is_invariant_to_scenario_order():
    s = np.array([[[30, 0, 0], [0, 0, 12], [0, 0, 10]], [[30, 0, 0], [20, 0, 5], [0, 0, 5]]], float)
    target = np.array([60, 30, 10], float)
    a = age_mpc(s, 35, reserve=target)
    b = age_mpc(s[::-1], 35, reserve=target)
    assert a == pytest.approx(b, abs=1e-6)


def test_joint_scenarios_never_exceed_generation_nameplate():
    from microgrid.trajectory_calibration import JointScenarioPool

    times = pd.date_range("2023-01-01", periods=60, freq="24h", tz="Asia/Kolkata")
    q = np.ones((60, 24, 3, 3))
    truth = np.ones((60, 24, 3))
    truth[:, :, 0] = 30
    truth[:, :, 1] = 15
    pool = JointScenarioPool(
        truth, q, times, np.array([10, 10, 1]), neighbors=90, generation_caps=[40, 20]
    )
    query = np.ones((24, 3, 3)) * 15
    query[:, 0, :] = 30
    sample = pool.sample(query, pd.Timestamp("2024-01-01", tz="Asia/Kolkata"))
    assert sample[:, :, 0].max() <= 40 and sample[:, :, 1].max() <= 20
    assert np.isfinite(sample).all() and (sample >= 0).all()


def test_exact_calendar_evaluator_matches_constant_stress_analytic():
    from microgrid.ageing import evaluate_lfp, graphite_potential

    soc = np.full(97, 0.5)
    k = 8.37e4 * np.exp(-5210 / 298.15) * np.exp(-3560 * graphite_potential(0.5) / 298.15)
    expected = float(k * 4**0.526)
    for window in [12, 24, 48]:
        result = evaluate_lfp(soc, chunk_hours=window, integration="effective_age_exact")
        assert result["calendar_loss_fraction"] == pytest.approx(expected, rel=1e-12)
        assert result["cycling_loss_fraction"] == 0


def test_low_temperature_charge_envelope_is_enforced():
    from microgrid.ageing import LfpState, update_lfp

    with pytest.raises(ValueError, match="0.16C"):
        update_lfp(LfpState(), [0, 1, 2], [0.1, 0.5, 0.9], [10, 10, 10])


def test_reachable_reserve_shortcut_matches_full_lexicographic_lp():
    rng = np.random.default_rng(66)
    for _ in range(8):
        s = rng.uniform(0, 25, (3, 8, 3))
        s[:, 0] = [30, 0, 2]
        c = BatteryConfig(capacity_kwh=400)
        target = np.full(8, 360.0)
        target[-1] = 40
        fast = age_mpc(s, 200, c, reserve=target)
        full = age_mpc(s, 200, c, reserve=target, force_full_lp=True)
        assert fast == pytest.approx(full, abs=5e-5)
