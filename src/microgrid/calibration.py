"""Held-out quantile interval correction; no exchangeability claim for time series."""

import numpy as np

from .data import CHANNELS
from .forecasting import metrics


class IntervalCalibrator:
    def __init__(self, alpha=0.2):
        if not 0 < alpha < 1:
            raise ValueError("Invalid interval alpha")
        self.alpha, self.correction, self.fit_end = alpha, None, None

    def fit(self, truth, prediction, fit_end):
        y, q = np.asarray(truth), np.asarray(prediction)
        if y.ndim != 3 or q.shape != (*y.shape, 3) or y.shape[-1] != 3:
            raise ValueError("Expected truth (origin,horizon,3), prediction (...,3 quantiles)")
        if not np.isfinite(y).all() or not np.isfinite(q).all() or (np.diff(q, axis=-1) < 0).any():
            raise ValueError("Nonfinite data or crossed quantiles")
        k = int(np.ceil((len(y) + 1) * (1 - self.alpha)))
        if k > len(y):
            raise ValueError("Too few calibration origins for requested finite-sample quantile")
        scores = np.maximum(q[..., 0] - y, y - q[..., 2])
        # Nonnegative widening only; one correction per lead and channel.
        self.correction = np.maximum(np.sort(scores, axis=0)[k - 1], 0)
        self.fit_end = int(fit_end)
        return self

    def apply(self, prediction, origin):
        if self.correction is None or origin < self.fit_end:
            raise ValueError("Unfitted calibrator or evaluation overlaps correction fit")
        q = np.asarray(prediction, dtype=float).copy()
        if q.shape != (*self.correction.shape, 3) or not np.isfinite(q).all():
            raise ValueError("Prediction shape or values invalid")
        q[..., 0] = np.maximum(0, q[..., 0] - self.correction)
        q[..., 2] += self.correction
        return q


def by_source_horizon(truth, prediction):
    y, q = np.asarray(truth), np.asarray(prediction)
    report = {"pooled": metrics(y, q), "by_source": {}, "by_horizon": {}}
    for channel, name in enumerate(CHANNELS):
        report["by_source"][name] = metrics(y[:, :, channel], q[:, :, channel])
    for h in range(y.shape[1]):
        report["by_horizon"][str(h + 1)] = {
            name: metrics(y[:, h, c], q[:, h, c]) for c, name in enumerate(CHANNELS)
        }
    # Use observed nonzero PV as a daylight-production diagnostic, not astronomy.
    mask = y[..., 0] > 1e-6
    report["pv_nonzero_observed"] = (
        metrics(y[..., 0][mask], q[..., 0, :][mask]) if mask.any() else None
    )
    return report
