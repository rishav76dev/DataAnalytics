"""Validation-budget selection for four SAC variants and three independent seeds.

Uses the same case/design as the separately frozen conventional benchmark.
No training decision reads final-test outcomes. Budget limits are disclosed.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from final_india_benchmark import dump, trace_metrics
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from microgrid.ageing import LfpState, update_lfp
from microgrid.data import CHANNELS, splits
from microgrid.environment import MicrogridEnv
from microgrid.forecasting import QuantileForecaster, residual_blocks
from microgrid.physics import BatteryConfig
from microgrid.reserve import vectorized_suffix_reserves
from microgrid.scenarios import sample_blocks


class StudyEnv(MicrogridEnv):
    """Adds published-model incremental damage proxy and observable ageing states."""

    def __init__(self, frame, config, reserve, variant, fixed_fraction, training=False):
        self.variant, self.training = variant, training
        self.age = LfpState()
        self.multiplier = 0.0
        if variant == "vanilla":
            reserve = None
        if variant == "fixed":
            reserve = np.full(len(frame), fixed_fraction * config.capacity)
        super().__init__(
            frame,
            config,
            reserve,
            reserve_weight=0.1 if variant != "vanilla" else 0,
            ageing_weight=0,
            episode_hours=min(168, len(frame)) if training else len(frame),
        )
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)

    def _observation(self):
        base = super()._observation()
        return np.array(
            [*base, self.age.calendar_loss, self.age.cycling_loss, self.age.efc / 100],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        self.age = LfpState()
        return super().reset(seed=seed, options={"random_start": self.training})

    def step(self, action):
        before = self.battery.soc
        obs, reward, done, truncated, info = super().step(action)
        after = self.battery.soc
        result = update_lfp(
            self.age, [0, 0.5, 1], [before, (before + after) / 2, after], [25, 25, 25]
        )
        # Assumed replacement value 100 currency/kWh, unserved value 1000/kWh.
        # This is a published-model-derived proxy, not calibrated economic cost.
        proxy = result["calendar_increment"] + result["cycling_increment"]
        reward -= 100 * self.config.capacity * proxy / 1000
        if self.variant == "constrained":
            reward -= self.multiplier * info["unserved_kw"] / 50
        info["published_ageing_proxy"] = proxy
        info["dual_multiplier"] = self.multiplier
        return self._observation(), reward, done, truncated, info


class DualUpdate(BaseCallback):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.unserved = 0.0
        self.demand = 0.0

    def _on_step(self):
        info = self.locals["infos"][0]
        self.unserved += info["unserved_kw"]
        self.demand += info["load_kw"]
        if self.locals["dones"][0]:
            if self.env.variant == "constrained":
                self.env.multiplier = max(
                    0, self.env.multiplier + 10 * (self.unserved / max(self.demand, 1e-9) - 0.01)
                )
            self.logger.record("constraint/multiplier", self.env.multiplier)
            self.unserved = 0.0
            self.demand = 0.0
        return True


def reserve_inputs(model, frame, residual, start, stop, c, seed=42):
    values = []
    for origin in range(start, stop):
        q = model.predict(frame, origin)
        s = sample_blocks(q, residual, count=100, seed=seed + origin)
        s[:, 0] = frame[CHANNELS].iloc[origin].to_numpy()
        targets, flags, _ = vectorized_suffix_reserves(s, c)
        values.append(np.nan if flags[1] else targets[1])
    return np.array(values)


def evaluate_policy(model, frame, c, reserve, variant, fixed, output):
    env = StudyEnv(frame, c, reserve, variant, fixed)
    obs, _ = env.reset()
    rows = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        rows.append(info)
        if done:
            break
    trace = pd.DataFrame(rows)
    trace["timestamp"] = frame.timestamp.to_numpy()
    trace.to_parquet(output.with_suffix(".parquet"), index=False)
    result = trace_metrics(trace, c)
    result["published_ageing_reward_proxy_total"] = float(trace.published_ageing_proxy.sum())
    dump(output.with_suffix(".json"), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["train", "evaluate"])
    p.add_argument("--data", default="data/india/reference/hourly.parquet")
    p.add_argument("--benchmark", default="outputs/india_study/final")
    p.add_argument("--output", default="outputs/india_study/rl")
    p.add_argument("--budgets", nargs="+", type=int, default=[10000, 30000])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = p.parse_args()
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    bench = Path(args.benchmark)
    frozen = json.loads((bench / "frozen_manifest.json").read_text())
    c = BatteryConfig(**frozen["battery"])
    fixed = frozen["fixed_fraction"]
    frame = pd.read_parquet(args.data)
    frame[["pv_kw", "wind_kw"]] *= frozen["selected_design"]["renewable_scale"]
    part = splits(len(frame))
    variants = ["vanilla", "fixed", "adaptive", "constrained"]
    if args.stage == "train":
        if (output / "selected.json").exists():
            raise ValueError("Selected run already exists")
        fit_end = int(0.6 * part["train"].stop)
        policy_start = int(0.8 * part["train"].stop)
        forecaster = QuantileForecaster(seed=42, trees=30).fit(frame, fit_end, stride=24)
        residual = residual_blocks(forecaster, frame, fit_end, policy_start)
        stop = part["validation"].stop
        reserves = reserve_inputs(forecaster, frame, residual, policy_start, stop, c)
        train_stop = part["train"].stop - policy_start
        training = frame.iloc[policy_start : part["train"].stop].reset_index(drop=True)
        validation = frame.iloc[part["validation"]].reset_index(drop=True)
        rtrain, rval = reserves[:train_stop], reserves[train_stop:]
        dump(
            output / "training_protocol.json",
            {
                "budgets": args.budgets,
                "seeds": args.seeds,
                "variants": variants,
                "fit_end": fit_end,
                "residual_stop": policy_start,
                "policy_training_start": policy_start,
                "policy_training_stop": part["train"].stop,
                "validation": [part["validation"].start, stop],
                "reward": "published-model hourly damage proxy + EENS + reserve; fixed25C; no capacity fade feedback",
                "observations": 12,
                "training_episode_hours": 168,
                "test_evaluated_during_selection": False,
            },
        )
        selected = {}
        for variant in variants:
            for seed in args.seeds:
                env = StudyEnv(training, c, rtrain, variant, fixed, training=True)
                monitor = Monitor(env, str(output / f"{variant}_{seed}_monitor.csv"))
                model = SAC(
                    "MlpPolicy",
                    monitor,
                    seed=seed,
                    device="cpu",
                    verbose=0,
                    learning_starts=500,
                    batch_size=64,
                    buffer_size=50000,
                    policy_kwargs={"net_arch": [64, 64]},
                )
                prior = 0
                candidates = []
                for budget in sorted(args.budgets):
                    model.learn(
                        total_timesteps=budget - prior,
                        callback=DualUpdate(env),
                        reset_num_timesteps=False,
                    )
                    checkpoint = output / f"{variant}_{seed}_{budget}"
                    model.save(checkpoint)
                    result = evaluate_policy(
                        model,
                        validation,
                        c,
                        rval,
                        variant,
                        fixed,
                        checkpoint.with_name(checkpoint.name + "_validation"),
                    )
                    candidates.append(
                        {
                            "budget": budget,
                            "checkpoint": str(checkpoint) + ".zip",
                            "eens_kwh": result["eens_kwh"],
                            "published_loss": result["published_model_ageing"][
                                "total_loss_fraction"
                            ],
                        }
                    )
                    prior = budget
                    print(
                        f"{variant} seed={seed} budget={budget} validation_EENS={result['eens_kwh']:.3f}",
                        flush=True,
                    )
                selected[f"{variant}_{seed}"] = min(
                    candidates, key=lambda r: (r["eens_kwh"], r["published_loss"])
                )
                dump(output / f"{variant}_{seed}_candidates.json", candidates)
                dump(output / "selected_partial.json", selected)
        # Frozen selection and hashes precede ANY RL final-test evaluation.
        import hashlib

        for entry in selected.values():
            entry["sha256"] = hashlib.sha256(Path(entry["checkpoint"]).read_bytes()).hexdigest()
        dump(output / "selected.json", selected)
    else:
        selected = json.loads((output / "selected.json").read_text())
        test = frame.iloc[part["test"]].reset_index(drop=True)
        rows = json.loads((bench / "reserves.json").read_text())
        if len(rows) != len(test):
            raise ValueError("Test reserve length mismatch")
        for row, stamp in zip(rows, test.timestamp):
            if pd.Timestamp(row["timestamp"]) != stamp:
                raise ValueError("Reserve timestamps mismatch")
        reserve = np.array([np.nan if r["end_infeasible"] else r["end_target_kwh"] for r in rows])
        results = {}
        for key, entry in selected.items():
            variant = key.rsplit("_", 1)[0]
            model = SAC.load(entry["checkpoint"], device="cpu")
            results[key] = evaluate_policy(
                model, test, c, reserve, variant, fixed, output / f"{key}_final"
            )
            print(
                f"{key} selected={entry['budget']} test_EENS={results[key]['eens_kwh']:.3f}",
                flush=True,
            )
        dump(output / "final_metrics.json", results)


if __name__ == "__main__":
    main()
