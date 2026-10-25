"""Independent LP checks of the fixed-physics, unweighted EENS exchange argument."""

import numpy as np
import pytest

from microgrid.data import synthetic
from microgrid.evaluation import dispatch
from microgrid.feasibility import perfect_foresight_episode
from microgrid.physics import BatteryConfig


def test_greedy_matches_eens_only_oracle_across_random_physics():
    rng = np.random.default_rng(74)
    for _ in range(60):
        frame = synthetic(24)
        frame[["pv_kw", "wind_kw", "load_kw"]] = rng.uniform(0, 60, (24, 3))
        c = BatteryConfig(
            capacity_kwh=float(rng.uniform(30, 300)),
            power_kw=float(rng.uniform(5, 60)),
            initial_soc=float(rng.uniform(0.1, 0.9)),
            soh=float(rng.uniform(0.6, 1)),
            charge_efficiency=float(rng.uniform(0.8, 1)),
            discharge_efficiency=float(rng.uniform(0.8, 1)),
            dt_hours=float(rng.choice([0.5, 1, 2])),
        )
        _, greedy = dispatch(frame, lambda t, e, x: x[2] - x[0] - x[1], c)
        _, oracle = perfect_foresight_episode(frame, c, terminal="minimum")
        assert greedy["eens_kwh"] == pytest.approx(oracle["eens_kwh"], abs=1e-7)
