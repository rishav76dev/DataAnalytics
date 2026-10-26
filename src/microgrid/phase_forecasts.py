"""Causal residual-to-weekly-baseline quantiles with local seasonal features."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import CHANNELS


def daylight(times, latitude=26.9157, longitude=70.9083):
    """NOAA solar geometry; retain any interval touching civil twilight (-6 deg)."""
    ts = pd.DatetimeIndex(times).tz_convert("UTC")
    result = np.zeros(len(ts), bool)
    for offset in [0, 0.25, 0.5, 0.75, 1]:
        t = ts + pd.Timedelta(hours=offset)
        hour = t.hour.to_numpy() + t.minute.to_numpy() / 60
        gamma = 2 * np.pi / 365 * (t.day_of_year.to_numpy() - 1 + (hour - 12) / 24)
        eq = 229.18 * (
            0.000075
            + 0.001868 * np.cos(gamma)
            - 0.032077 * np.sin(gamma)
            - 0.014615 * np.cos(2 * gamma)
            - 0.040849 * np.sin(2 * gamma)
        )
        decl = (
            0.006918
            - 0.399912 * np.cos(gamma)
            + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma)
            + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma)
            + 0.00148 * np.sin(3 * gamma)
        )
        angle = np.deg2rad((hour * 60 + eq + 4 * longitude) % 1440 / 4 - 180)
        cosine = np.sin(np.deg2rad(latitude)) * np.sin(decl) + np.cos(
            np.deg2rad(latitude)
        ) * np.cos(decl) * np.cos(angle)
        result |= cosine >= np.sin(np.deg2rad(-6))
    return result


def feature_matrix(frame, origins, horizon=24):
    origins = np.asarray(origins, int)
    if (
        len(origins) == 0
        or origins.min() < 168
        or origins.max() > len(frame)
        or not 1 <= horizon <= 24
    ):
        raise ValueError("Insufficient history or invalid origins/horizon")
    vals = frame[CHANNELS].to_numpy(float)
    h = np.arange(horizon)
    baseline = vals[origins[:, None] - 168 + h]
    times = pd.DatetimeIndex(frame.timestamp.iloc[origins - 1]).repeat(horizon) + pd.to_timedelta(
        np.tile(h + 1, len(origins)), unit="h"
    )
    local = times.tz_convert("Asia/Kolkata")
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    yearangle = 2 * np.pi * (local.day_of_year.to_numpy() - 1) / 365.25
    cols = [
        np.tile(h + 1, len(origins)),
        np.sin(hour * np.pi / 12),
        np.cos(hour * np.pi / 12),
        local.day_of_week.to_numpy(),
        np.sin(yearangle),
        np.cos(yearangle),
    ]
    sums = np.vstack([np.zeros(3), np.cumsum(vals, axis=0)])
    squares = np.vstack([np.zeros(3), np.cumsum(vals**2, axis=0)])
    mean = (sums[origins] - sums[origins - 24]) / 24
    variance = np.maximum((squares[origins] - squares[origins - 24]) / 24 - mean**2, 0)
    for c in range(3):
        for lag in [1, 24, 168]:
            cols.append(np.repeat(vals[origins - lag, c], horizon))
        cols.extend(
            [
                np.repeat(mean[:, c], horizon),
                np.repeat(np.sqrt(variance[:, c]), horizon),
                baseline[:, :, c].ravel(),
            ]
        )
    return np.column_stack(cols), baseline, times


class ResearchForecaster:
    def __init__(self, seed=42, trees=120, generation_caps=None):
        self.seed, self.trees = seed, trees
        if generation_caps is None:
            raise ValueError("Explicit PV/wind nameplate capacities are required")
        self.generation_caps = np.asarray(generation_caps, float)
        if (
            self.generation_caps.shape != (2,)
            or not np.isfinite(self.generation_caps).all()
            or (self.generation_caps <= 0).any()
        ):
            raise ValueError("Invalid PV/wind nameplate capacities")
        self.models = {}
        self.training_end = None

    def fit(self, frame, stop, stride=6):
        if (frame[CHANNELS].iloc[:stop].to_numpy()[:, :2] > self.generation_caps + 1e-6).any():
            raise ValueError("Training generation exceeds nameplate")
        origins = np.arange(168, stop - 24 + 1, stride)
        x, base, _ = feature_matrix(frame, origins)
        idx = origins[:, None] + np.arange(24)
        target = frame[CHANNELS].to_numpy()[idx] - base
        for c in range(3):
            for k, q in enumerate([0.1, 0.5, 0.9]):
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=self.trees,
                    num_leaves=23,
                    min_child_samples=40,
                    verbosity=-1,
                    n_jobs=2,
                    random_state=self.seed,
                    deterministic=True,
                    force_col_wise=True,
                )
                model.fit(x, target[:, :, c].ravel())
                self.models[c, k] = model
        self.training_end = int(stop)
        self.scale_floor = np.maximum(frame[CHANNELS].iloc[:stop].std().to_numpy(), 0.05)
        return self

    def predict_many(self, frame, origins):
        origins = np.asarray(origins, int)
        if self.training_end is None or origins.min() < self.training_end:
            raise ValueError("Forecast overlaps training")
        results = []
        for start in range(0, len(origins), 512):
            x, base, times = feature_matrix(frame, origins[start : start + 512])
            out = np.empty((*base.shape, 3))
            for c in range(3):
                for k in range(3):
                    out[:, :, c, k] = (
                        self.models[c, k].booster_.predict(x, num_threads=2).reshape(-1, 24)
                        + base[:, :, c]
                    )
            out = np.sort(np.maximum(out, 0), axis=-1)
            out[:, :, :2, :] = np.minimum(
                out[:, :, :2, :], self.generation_caps[None, None, :, None]
            )
            out[:, :, 0, :] *= daylight(times).reshape(-1, 24, 1)
            results.append(out)
        return np.concatenate(results)
