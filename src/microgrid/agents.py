from pathlib import Path

import numpy as np

from .environment import MicrogridEnv
from .physics import BatteryConfig


class ConstrainedSAC:
    """Episode-wise Lagrangian SAC wrapper; reliability is not a hard guarantee.

    A dual multiplier penalizes normalized unserved energy. Physical limits remain
    enforced by the environment. Evaluate constraint satisfaction out of sample.
    """

    def __init__(self, env, target_lpsp=0.01, dual_lr=10.0, seed=42):
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback

        self.multiplier = 0.0
        self.target_lpsp, self.dual_lr = target_lpsp, dual_lr
        owner = self

        class ReliabilityPenalty(MicrogridEnv):
            def step(self, action):
                obs, reward, done, truncated, info = super().step(action)
                info["dual_cost"] = info["unserved_kw"] / self.power_scale
                return obs, reward - owner.multiplier * info["dual_cost"], done, truncated, info

        class DualUpdate(BaseCallback):
            def __init__(self):
                super().__init__()
                self.unserved, self.demand = 0.0, 0.0

            def _on_step(self):
                info = self.locals["infos"][0]
                self.unserved += info["unserved_kw"]
                self.demand += info["load_kw"]
                if self.locals["dones"][0]:
                    lpsp = self.unserved / max(self.demand, 1e-8)
                    owner.multiplier = max(
                        0, owner.multiplier + owner.dual_lr * (lpsp - owner.target_lpsp)
                    )
                    self.logger.record("constraint/multiplier", owner.multiplier)
                    self.logger.record("constraint/lpsp", lpsp)
                    self.unserved, self.demand = 0.0, 0.0
                return True

        wrapped = ReliabilityPenalty(
            env.frame,
            env.config,
            env.reserve,
            env.reserve_weight,
            env.ageing_weight,
            env.power_scale,
            env.episode_hours,
        )
        self.model = SAC(
            "MlpPolicy",
            wrapped,
            seed=seed,
            device="cpu",
            verbose=0,
            learning_starts=100,
            batch_size=64,
            buffer_size=50000,
            policy_kwargs={"net_arch": [64, 64]},
        )
        self.callback = DualUpdate()

    def learn(self, steps):
        self.model.learn(steps, callback=self.callback)
        return self


def train(
    frame, output, variant="vanilla", steps=2000, seed=42, reserve=None, config=BatteryConfig()
):
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    torch.set_num_threads(2)
    if variant not in {"vanilla", "fixed", "adaptive", "constrained"}:
        raise ValueError("Unknown agent variant")
    if variant in {"adaptive", "constrained"} and reserve is None:
        raise ValueError("Adaptive variants require causally computed reserves")
    if variant == "fixed":
        reserve = np.full(len(frame), 0.4 * config.capacity)
    env = MicrogridEnv(
        frame, config, reserve=reserve, reserve_weight=0.1 if variant != "vanilla" else 0
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if variant == "constrained":
        agent = ConstrainedSAC(env, seed=seed).learn(steps)
        model = agent.model
        output.with_suffix(".dual.txt").write_text(str(agent.multiplier))
    else:
        monitored = Monitor(env, filename=str(output) + ".monitor.csv")
        model = SAC(
            "MlpPolicy",
            monitored,
            seed=seed,
            device="cpu",
            verbose=0,
            learning_starts=100,
            batch_size=64,
            buffer_size=50000,
            policy_kwargs={"net_arch": [64, 64]},
        )
        model.learn(total_timesteps=steps)
    model.save(output)
    return model


def training_inputs(frame, variant, seed=42):
    """Nested chronological fit/calibration/policy-training inside outer train only.

    Use the identical policy-training suffix for all variants. No validation,
    outer calibration, or test outcomes are used to construct training reserves.
    """
    from .forecasting import QuantileForecaster, residual_blocks
    from .reserve import ReserveEstimator
    from .scenarios import sample_blocks

    n = len(frame)
    fit_end, policy_start = int(0.6 * n), int(0.8 * n)
    if fit_end < 220 or policy_start - fit_end < 48:
        raise ValueError("Need more data for causal nested RL training (>=720 outer-train hours)")
    if variant in {"adaptive", "constrained"}:
        forecaster = QuantileForecaster(seed=seed, trees=30).fit(frame, fit_end)
        residual = residual_blocks(forecaster, frame, fit_end, policy_start)
        estimator, reserves = ReserveEstimator(), []
        for origin in range(policy_start, n):
            q = forecaster.predict(frame, origin)
            scenarios = sample_blocks(q, residual, count=100, seed=seed + origin)
            # End-of-current-step reserve: next 23 hours, known only as forecasts.
            result = estimator.estimate(scenarios[:, 1:])
            reserves.append(result["target_kwh"] if result["risk_feasible"] else np.nan)
        reserve = np.asarray(reserves)
    else:
        reserve = None
    return frame.iloc[policy_start:].reset_index(drop=True), reserve


def evaluate(model_path, frame, output, variant="vanilla", reserve=None, config=BatteryConfig()):
    import json

    import pandas as pd
    import torch
    from stable_baselines3 import SAC

    from .evaluation import degradation, reliability

    torch.set_num_threads(2)
    if variant == "fixed":
        reserve = np.full(len(frame), 0.4 * config.capacity)
    if variant in {"adaptive", "constrained"} and reserve is None:
        raise ValueError("Need causal evaluation reserves")
    env = MicrogridEnv(frame, config, reserve, 0.1 if variant != "vanilla" else 0)
    model = SAC.load(model_path, device="cpu")
    obs, _ = env.reset()
    rows = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        rows.append(info)
        if done:
            break
    trace = pd.DataFrame(rows)
    result = reliability(trace, config.dt_hours)
    result["ageing"] = degradation([config.initial_soc, *trace.soc], len(trace))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(result, indent=2))
    trace.to_parquet(output.with_suffix(".parquet"), index=False)
    return result
