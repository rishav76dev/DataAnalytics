import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data import CHANNELS, audit
from .physics import Battery, BatteryConfig


class MicrogridEnv(gym.Env):
    """Causal hourly environment. Current bus powers are measured at dispatch.

    No future measurements in observation. Reserves must be precomputed causally;
    NaN reserve values denote forecast infeasibility, never a physical guarantee.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        frame,
        config=BatteryConfig(),
        reserve=None,
        reserve_weight=0.0,
        ageing_weight=0.01,
        power_scale=50.0,
        episode_hours=None,
    ):
        super().__init__()
        audit(frame)
        self.frame, self.config = frame.reset_index(drop=True), config
        self.values = self.frame[CHANNELS].to_numpy(dtype=float)
        self.reserve = np.zeros(len(frame)) if reserve is None else np.asarray(reserve, dtype=float)
        if self.reserve.shape != (len(frame),) or np.isinf(self.reserve).any():
            raise ValueError("Reserve length mismatch or infinite reserve; use NaN for infeasible")
        if (np.nan_to_num(self.reserve) < 0).any() or power_scale <= 0:
            raise ValueError("Invalid reserve or scale")
        self.reserve_weight, self.ageing_weight = reserve_weight, ageing_weight
        self.power_scale = power_scale
        self.episode_hours = len(frame) if episode_hours is None else int(episode_hours)
        if not 1 <= self.episode_hours <= len(frame):
            raise ValueError("Invalid episode length")
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
        self.reset()

    def _observation(self):
        i = min(self.t, len(self.frame) - 1)
        hour = self.frame.timestamp.iloc[i].hour
        target = self.reserve[i]
        return np.array(
            [
                self.battery.soc,
                *self.values[i] / self.power_scale,
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.nan_to_num(target) / self.config.capacity,
                float(np.isnan(target)),
                self.config.soh,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        random_start = (options or {}).get("random_start", False)
        self.t = (
            int(self.np_random.integers(len(self.frame) - self.episode_hours + 1))
            if random_start
            else 0
        )
        self.end = self.t + self.episode_hours
        self.battery = Battery(self.config)
        self.finished = False
        return self._observation(), {}

    def step(self, action):
        if self.finished:
            raise RuntimeError("Call reset after episode termination")
        a = np.asarray(action)
        if a.shape != (1,) or not np.isfinite(a).all():
            raise ValueError("Action must be a finite shape-(1,) array")
        info = self.battery.step(
            float(np.clip(a[0], -1, 1)) * self.config.power_kw, *self.values[self.t]
        )
        target = self.reserve[self.t]
        deficit = max(target - info["energy_kwh"], 0) if np.isfinite(target) else 0
        # Reward scaled so EENS has dominant cost. Units documented in protocol.
        cost = (
            1000 * info["unserved_kw"] * self.config.dt_hours
            + self.ageing_weight * info["throughput_kwh"]
            + 0.001 * info["curtailment_kw"] * self.config.dt_hours
            + self.reserve_weight * deficit
        )
        info["reserve_deficit_kwh"] = deficit
        info["reserve_infeasible"] = bool(np.isnan(target))
        self.t += 1
        self.finished = self.t >= self.end
        return self._observation(), -float(cost) / 1000, self.finished, False, info
