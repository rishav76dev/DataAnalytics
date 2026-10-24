import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from microgrid.data import synthetic
from microgrid.environment import MicrogridEnv
from microgrid.physics import Battery, BatteryConfig


def test_random_dispatch_conserves_energy():
    c = BatteryConfig(soh=0.8)
    b = Battery(c)
    rng = np.random.default_rng(5)
    for _ in range(1000):
        pv, wind, load = rng.uniform(0, 80, 3)
        before = b.energy
        r = b.step(rng.uniform(-150, 150), pv, wind, load)
        p = r["battery_kw"]
        assert pv + wind + p + r["unserved_kw"] == pytest.approx(load + r["curtailment_kw"])
        assert c.minimum - 1e-9 <= b.energy <= c.maximum + 1e-9
        assert abs(p) <= c.power_kw
        assert b.energy - before == pytest.approx(
            -p / c.discharge_efficiency if p >= 0 else -p * c.charge_efficiency
        )


def test_limits_efficiency_and_no_grid_charging():
    b = Battery()
    b.step(-50, 60, 0, 0)
    assert b.energy == 90
    assert b.step(-50, 0, 0, 10)["battery_kw"] == 0
    b.step(50, 0, 0, 80)
    assert b.energy == pytest.approx(90 - 50 / 0.95)
    b.step(50, 0, 0, 80)
    assert b.energy == pytest.approx(10)


def test_gym_contract_and_termination():
    env = MicrogridEnv(synthetic(24))
    check_env(env)
    env.reset(seed=1)
    for _ in range(24):
        obs, reward, done, truncated, info = env.step(np.array([0.5], np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward)
    assert done and not truncated
    with pytest.raises(RuntimeError):
        env.step([0])


@pytest.mark.parametrize(
    "kwargs", [{"soh": 0}, {"power_kw": -1}, {"soc_min": 0.9}, {"dt_hours": 0}]
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        BatteryConfig(**kwargs)
