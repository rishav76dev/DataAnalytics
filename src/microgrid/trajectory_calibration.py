"""Phase/season-aware joint scenarios and causal simultaneous deficit envelopes.

The online envelope targets empirical whole-path coverage, not calibrated
probabilities for each bootstrap draw. No exchangeable guarantee is asserted.
"""

from collections import deque

import numpy as np
import pandas as pd

from .phase_forecasts import daylight


def dc_flow(values, charge_efficiency=0.95, discharge_efficiency=0.95, power_kw=50.0):
    net = values[..., 2] - values[..., 0] - values[..., 1]
    return (
        np.maximum(net, 0) / discharge_efficiency
        - np.minimum(np.maximum(-net, 0), power_kw) * charge_efficiency
    )


def intervals(flow):
    h = flow.shape[-1]
    i, j = np.triu_indices(h)
    prefix = np.concatenate([np.zeros((*flow.shape[:-1], 1)), np.cumsum(flow, axis=-1)], axis=-1)
    return prefix[..., j + 1] - prefix[..., i], i, j


class JointScenarioPool:
    def __init__(self, truth, quantiles, times, scale_floor, neighbors=90, generation_caps=None):
        self.floor = np.asarray(scale_floor)
        if generation_caps is None:
            raise ValueError("Explicit PV/wind nameplate capacities are required")
        self.generation_caps = np.asarray(generation_caps, float)
        if (
            self.generation_caps.shape != (2,)
            or not np.isfinite(self.generation_caps).all()
            or (self.generation_caps <= 0).any()
        ):
            raise ValueError("Invalid PV/wind nameplate capacities")
        if (np.asarray(truth)[..., :2] > self.generation_caps + 1e-6).any():
            raise ValueError("Historical generation exceeds nameplate")
        self.residual = (truth - quantiles[..., 1]) / np.maximum(
            np.abs(quantiles[..., 1]) * 0.25, self.floor
        )
        stamp = pd.DatetimeIndex(times).tz_convert("Asia/Kolkata")
        self.hour = stamp.hour.to_numpy()
        self.day = stamp.day_of_year.to_numpy()
        self.end = stamp.max() + pd.Timedelta(hours=24)
        self.neighbors = neighbors

    def sample(self, q, stamp, count=100, seed=42, independent=False):
        stamp = pd.Timestamp(stamp).tz_convert("Asia/Kolkata")
        if stamp < self.end:
            raise ValueError("Scenario inference overlaps unfinished residual trajectories")
        idx = np.flatnonzero(self.hour == stamp.hour)
        distance = np.minimum(
            np.abs(self.day[idx] - stamp.day_of_year),
            366 - np.abs(self.day[idx] - stamp.day_of_year),
        )
        idx = idx[np.argsort(distance, kind="stable")[: self.neighbors]]
        if len(idx) < 30:
            raise ValueError("Insufficient phase-matched historical days")
        rng = np.random.default_rng(seed)
        if independent:
            chosen = rng.choice(idx, size=(count, 24, 3))
            error = self.residual[chosen, np.arange(24)[None, :, None], np.arange(3)[None, None, :]]
        else:
            error = self.residual[rng.choice(idx, size=count)]
        out = np.maximum(
            q[None, :, :, 1] + error * np.maximum(np.abs(q[:, :, 1]) * 0.25, self.floor), 0
        )
        times = pd.date_range(stamp, periods=24, freq="h")
        out[:, :, 0] *= daylight(times)[None, :]
        out[:, :, :2] = np.minimum(out[:, :, :2], self.generation_caps[None, None, :])
        return out


class OnlineDeficitEnvelope:
    def __init__(self, alpha=0.05, window_days=90, min_days=30):
        if not 0 < alpha < 1 or window_days < min_days or min_days < 20:
            raise ValueError("Invalid calibration settings")
        self.alpha = alpha
        self.min_days = min_days
        self.scores = [deque(maxlen=window_days) for _ in range(24)]
        self.pending = {}

    def advance(self, origin, frame):
        """Only use trajectories whose last observation is strictly before origin."""
        for old in sorted(list(self.pending)):
            if old + 24 > origin:
                continue
            bounds, i, j, phase = self.pending.pop(old)
            actual = frame.iloc[old + 1 : old + 24][["pv_kw", "wind_kw", "load_kw"]].to_numpy()
            totals, _, _ = intervals(dc_flow(actual))
            score = float(np.max((totals - bounds) / np.sqrt(j - i + 1)))
            self.scores[phase].append(score)

    def predict(self, origin, stamp, scenarios, capacity_kwh, min_energy, max_energy):
        flow = dc_flow(scenarios[:, 1:])
        total, i, j = intervals(flow)
        raw = np.quantile(total, 1 - self.alpha, axis=0, method="higher")
        phase = pd.Timestamp(stamp).tz_convert("Asia/Kolkata").hour
        history = self.scores[phase]
        k = int(np.ceil((len(history) + 1) * (1 - self.alpha)))
        ready = len(history) >= self.min_days and k <= len(history)
        correction = max(float(np.sort(history)[k - 1]), 0) if ready else None
        self.pending[origin] = (raw, i, j, phase)
        bound = raw + (correction or 0) * np.sqrt(j - i + 1)
        targets = []
        flags = []
        needed_values = []
        maximum_values = []
        for start in range(23):
            needed = max(0.0, float(np.max(bound[i == start])))
            needed_values.append(needed)
            maximum_values.append(max(0.0, float(np.max(bound[i >= start]))))
            infeasible = (not ready) or bool(
                np.max(bound[i >= start]) > max_energy - min_energy + 1e-8
            )
            targets.append(max_energy if infeasible else min_energy + needed)
            flags.append(infeasible)
        targets.append(min_energy)
        flags.append(False)
        return np.asarray(targets), {
            "ready": ready,
            "correction_dc_kwh_per_sqrt_hour": correction,
            "calibration_days": len(history),
            "risk_infeasible": flags[0],
            "usable_targets_dc_kwh": [*needed_values, 0.0],
            "max_interval_deficit_dc_kwh": [*maximum_values, 0.0],
            "raw_bounds": raw,
            "corrected_bounds": bound,
            "interval_i": i,
            "interval_j": j,
            "suffix_infeasible": flags,
            "capacity_kwh": capacity_kwh,
        }

    @staticmethod
    def coverage(actual, diagnostic):
        totals, _, _ = intervals(dc_flow(actual[1:]))
        return {
            "raw_joint_path_covered": bool(np.all(totals <= diagnostic["raw_bounds"] + 1e-8)),
            "corrected_joint_path_covered": bool(
                np.all(totals <= diagnostic["corrected_bounds"] + 1e-8)
            ),
            "raw_net_energy_covered": bool(totals[22] <= diagnostic["raw_bounds"][22] + 1e-8),
        }
