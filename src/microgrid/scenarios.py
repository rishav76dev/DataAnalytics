import numpy as np


def sample_blocks(quantiles, residuals, count=100, seed=42, independent=False):
    """Bootstrap whole calibrated trajectories, synchronously across all channels.

    Residuals are additive P50 errors. Quantile intervals are evaluated separately;
    this empirical bootstrap does not assert calibrated tail probabilities.
    """
    q, r = np.asarray(quantiles), np.asarray(residuals)
    if q.ndim != 3 or q.shape[-2:] != (3, 3):
        raise ValueError("Expected (horizon, channel, quantile)")
    if r.ndim != 3 or r.shape[1:] != q.shape[:2] or len(r) < 2 or count < 1:
        raise ValueError("Residual block shape/count mismatch")
    if not np.isfinite(q).all() or not np.isfinite(r).all():
        raise ValueError("Nonfinite sampler inputs")
    rng = np.random.default_rng(seed)
    if independent:
        # Deliberately destroys both temporal and cross-source dependence for ablation.
        idx = rng.integers(len(r), size=(count, *q.shape[:2]))
        sampled = r[idx, np.arange(q.shape[0])[None, :, None], np.arange(3)[None, None, :]]
    else:
        sampled = r[rng.integers(len(r), size=count)]
    return np.maximum(q[None, :, :, 1] + sampled, 0)
