import numpy as np
import pytest

from microgrid.controllers import deterministic_mpc
from microgrid.data import synthetic
from microgrid.environment import MicrogridEnv
from microgrid.reserve import (
    ReserveEstimator,
    backward_requirement,
    end_of_step_targets,
    optimization_oracle,
)


def test_low_deficit_reserves_and_suffix_alignment():
    rng = np.random.default_rng(91)
    for _ in range(80):
        net = rng.uniform(-10, 8, 24)
        a, b = backward_requirement(net), optimization_oracle(net)
        assert a.feasible == b.feasible
        if a.feasible:
            assert a.energy_kwh == pytest.approx(b.energy_kwh)
    scenarios = np.zeros((10, 3, 3))
    scenarios[:, :, 2] = [10, 20, 30]
    targets, bad = end_of_step_targets(scenarios, ReserveEstimator())
    assert targets == pytest.approx([10 + 50 / 0.95, 10 + 30 / 0.95, 10])
    assert not any(bad)
    assert deterministic_mpc(scenarios[0], 90, reserve_kwh=targets) == pytest.approx(10)


def test_reserve_infeasibility_is_observable():
    env = MicrogridEnv(synthetic(24), reserve=np.full(24, np.nan), reserve_weight=0.1)
    obs, _ = env.reset()
    assert obs[-2] == 1
    _, reward, _, _, info = env.step(np.array([0], np.float32))
    assert np.isfinite(reward) and info["reserve_infeasible"]


def test_sac_checkpoint_and_reload(tmp_path):
    pytest.importorskip("stable_baselines3")
    from microgrid.agents import evaluate, train

    frame = synthetic(24)
    path = tmp_path / "policy"
    train(frame, path, steps=110, seed=3)
    result = evaluate(path, frame, tmp_path / "eval")
    assert result["demand_kwh"] == pytest.approx(frame.load_kw.sum())
    assert path.with_suffix(".zip").exists()
