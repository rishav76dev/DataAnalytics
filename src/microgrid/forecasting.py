import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import CHANNELS

QUANTILES = (0.1, 0.5, 0.9)


def features(frame, origin, horizon):
    """Origin is the first unknown row. Every measured feature ends before origin."""
    if origin < 168 or not 1 <= horizon <= 24:
        raise ValueError("Need 168 past hours and horizon in [1,24]")
    past = frame.iloc[:origin]
    timestamp = pd.Timestamp(past.timestamp.iloc[-1]) + pd.Timedelta(hours=horizon)
    row = [
        horizon,
        np.sin(2 * np.pi * timestamp.hour / 24),
        np.cos(2 * np.pi * timestamp.hour / 24),
        timestamp.day_of_week,
    ]
    for c in CHANNELS:
        values = past[c].to_numpy()
        row += [
            values[-1],
            values[-24],
            values[-168],
            values[-24:].mean(),
            values[-24:].std(),
            values[-168 + (horizon - 1)],
        ]
    return np.asarray(row, dtype=float)


class QuantileForecaster:
    """Pooled direct horizon-aware LightGBM models: one model per channel/quantile."""

    def __init__(self, seed=42, trees=60):
        self.seed, self.trees, self.models = seed, trees, {}
        self.training_end = None

    def fit(self, frame, stop, stride=6):
        if stop > len(frame) or stop < 168 + 24 + 1:
            raise ValueError("Insufficient training history")
        samples, targets = [], []
        for origin in range(168, stop - 24 + 1, stride):
            for h in range(1, 25):
                samples.append(features(frame, origin, h))
                targets.append(frame[CHANNELS].iloc[origin + h - 1].to_numpy())
        x, y = np.asarray(samples), np.asarray(targets)
        for c in range(3):
            for q in QUANTILES:
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    n_estimators=self.trees,
                    num_leaves=15,
                    min_child_samples=25,
                    verbosity=-1,
                    random_state=self.seed,
                    n_jobs=2,
                    deterministic=True,
                    force_col_wise=True,
                )
                model.fit(x, y[:, c])
                self.models[c, q] = model
        self.training_end = stop
        return self

    def predict(self, frame, origin, horizon=24):
        if not self.models or not 1 <= horizon <= 24:
            raise ValueError("Model not fitted or invalid horizon")
        if origin < self.training_end:
            raise ValueError("Evaluation origin overlaps training")
        x = np.array([features(frame, origin, h) for h in range(1, horizon + 1)])
        out = np.empty((horizon, 3, 3))
        for c in range(3):
            for k, q in enumerate(QUANTILES):
                out[:, c, k] = self.models[c, q].booster_.predict(x, num_threads=2)
        return np.sort(np.maximum(out, 0), axis=-1)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(pickle.dumps(self))


def residual_blocks(model, frame, start, stop, stride=24):
    """Use only completed out-of-training rolling origins in calibration interval."""
    if start < model.training_end or stop > len(frame):
        raise ValueError("Calibration must follow training and be observed")
    blocks = []
    for origin in range(start, stop - 24 + 1, stride):
        prediction = model.predict(frame, origin)
        blocks.append(frame[CHANNELS].iloc[origin : origin + 24].to_numpy() - prediction[:, :, 1])
    if len(blocks) < 2:
        raise ValueError("Need at least two completed calibration blocks")
    return np.asarray(blocks)


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    error = truth[..., None] - prediction
    point_error = error[..., 1]
    q = np.array(QUANTILES)
    return {
        "mae_kw": float(np.abs(point_error).mean()),
        "rmse_kw": float(np.sqrt((point_error**2).mean())),
        "pinball": float(np.maximum(q * error, (q - 1) * error).mean()),
        "coverage_80": float(
            ((truth >= prediction[..., 0]) & (truth <= prediction[..., 2])).mean()
        ),
        "interval_width_kw": float((prediction[..., 2] - prediction[..., 0]).mean()),
    }
