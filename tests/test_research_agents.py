from pathlib import Path

import numpy as np
import pytest

from microgrid.data import synthetic
from microgrid.physics import BatteryConfig


def test_published_proxy_environment_state_and_episode_reset(monkeypatch):
    pytest.importorskip("stable_baselines3")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from india_rl_study import StudyEnv

    frame = synthetic(72)
    env = StudyEnv(frame, BatteryConfig(capacity_kwh=200), np.full(72, 50.0), "adaptive", 0.4)
    obs, _ = env.reset(seed=4)
    assert obs.shape == (12,) and np.isfinite(obs).all()
    for _ in range(24):
        obs, reward, done, _, info = env.step(np.array([0.5], np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        assert info["published_ageing_proxy"] >= 0
        assert 0.1 - 1e-9 <= env.battery.soc <= 0.9 + 1e-9
    assert env.age.calendar_loss > 0 and env.age.efc > 0
    env.reset()
    assert env.age.soh == 1 and env.age.efc == 0
