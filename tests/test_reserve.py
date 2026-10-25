import numpy as np
import pytest

from microgrid.physics import BatteryConfig
from microgrid.reserve import ReserveEstimator, backward_requirement, optimization_oracle


def test_recursion_matches_independent_lp_randomized():
    rng = np.random.default_rng(4)
    for c in [BatteryConfig(), BatteryConfig(soh=0.7, dt_hours=0.5)]:
        for _ in range(120):
            net = rng.uniform(-65, 55, 24)
            a, b = backward_requirement(net, c), optimization_oracle(net, c)
            assert a.feasible == b.feasible
            if a.feasible:
                assert a.energy_kwh == pytest.approx(b.energy_kwh)


def test_order_power_and_energy_infeasibility():
    assert backward_requirement([-50, 50]).energy_kwh == pytest.approx(10 + 50 / 0.95 - 47.5)
    assert backward_requirement([50, -50]).energy_kwh == pytest.approx(10 + 50 / 0.95)
    assert backward_requirement([51]).reason == "discharge_power"
    assert backward_requirement([40, 40]).reason == "energy_capacity"


def test_infeasible_mass_not_conditioned_away():
    s = np.zeros((100, 24, 3))
    s[:6, :, 2] = 60
    r = ReserveEstimator().estimate(s)
    assert r["target_kwh"] is None and not r["risk_feasible"]
    assert r["infeasible_fraction"] == 0.06
    s[5, :, 2] = 0
    r = ReserveEstimator().estimate(s)
    assert r["risk_feasible"] and r["target_kwh"] == 10


def test_vectorized_all_suffixes_match_original_including_infeasibility():
    from microgrid.reserve import vectorized_suffix_reserves

    rng = np.random.default_rng(2026)
    for config in [BatteryConfig(), BatteryConfig(soh=0.7, dt_hours=0.5)]:
        scenarios = rng.uniform(0, 40, (37, 24, 3))
        target, flags, fraction = vectorized_suffix_reserves(scenarios, config)
        for t in range(24):
            expected = ReserveEstimator(config).estimate(scenarios[:, t:])
            assert bool(flags[t]) == (expected["target_kwh"] is None)
            assert fraction[t] == pytest.approx(expected["infeasible_fraction"])
            if not flags[t]:
                assert target[t] == pytest.approx(expected["target_kwh"])
        assert target[-1] == config.minimum and not flags[-1]
