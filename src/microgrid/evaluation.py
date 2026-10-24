import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rainflow

from .data import CHANNELS
from .physics import Battery, BatteryConfig


@dataclass(frozen=True)
class AgeingParameters:
    """Illustrative sensitivity parameters; NOT a validated empirical LFP model."""

    reference_cycles: float = 4000
    reference_depth: float = 0.8
    cycle_exponent: float = 1.5
    calendar_loss_per_year: float = 0.02


def degradation(soc, hours, params=AgeingParameters()):
    soc = np.asarray(soc, dtype=float)
    if not np.isfinite(soc).all() or (soc < 0).any() or (soc > 1).any() or hours < 0:
        raise ValueError("Invalid SOC history or duration")
    cycles = list(rainflow.extract_cycles(soc))
    # reference_cycles to 80% SOH at reference depth: 0.2 total capacity loss.
    cyclic = sum(
        0.2
        / params.reference_cycles
        * (depth / params.reference_depth) ** params.cycle_exponent
        * count
        for depth, _, count, _, _ in cycles
    )
    calendar = params.calendar_loss_per_year * hours / (365.25 * 24)
    return {
        "model_status": "illustrative_unvalidated",
        "cyclic_loss_fraction": cyclic,
        "calendar_loss_fraction": calendar,
        "total_loss_fraction": cyclic + calendar,
        "equivalent_full_cycles": sum(d * count for d, _, count, _, _ in cycles),
    }


def reliability(trace, dt=1):
    unserved = trace.unserved_kw.to_numpy()
    demand = float(trace.load_kw.sum() * dt)
    return {
        "eens_kwh": float(unserved.sum() * dt),
        "lpsp_energy": float(unserved.sum() * dt / demand) if demand else 0.0,
        "loss_of_load_hours": float((unserved > 1e-7).sum() * dt),
        "curtailment_kwh": float(trace.curtailment_kw.sum() * dt),
        "throughput_kwh": float(trace.throughput_kwh.sum()),
        "demand_kwh": demand,
    }


def dispatch(frame, controller, config=BatteryConfig()):
    battery = Battery(config)
    rows, durations = [], []
    for t, x in enumerate(frame[CHANNELS].to_numpy()):
        start = time.perf_counter()
        action = controller(t, battery.energy, x)
        durations.append(time.perf_counter() - start)
        row = battery.step(action, *x)
        row["timestamp"] = frame.timestamp.iloc[t]
        rows.append(row)
    trace = pd.DataFrame(rows)
    result = reliability(trace, config.dt_hours)
    result.update(
        {
            "controller_seconds": sum(durations),
            "controller_p95_ms": float(np.quantile(durations, 0.95) * 1000),
            "ageing": degradation([config.initial_soc, *trace.soc], len(trace) * config.dt_hours),
        }
    )
    return trace, result


def paired_block_bootstrap(a, b, block=24, repeats=1000, seed=42):
    """CI for paired hourly metric difference, with circular chronological blocks."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < block or block < 1 or repeats < 1:
        raise ValueError("Invalid paired series or bootstrap settings")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite bootstrap series")
    rng, delta = np.random.default_rng(seed), a - b
    n = len(delta)
    means = []
    for _ in range(repeats):
        starts = rng.integers(n, size=int(np.ceil(n / block)))
        idx = ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
        means.append(delta[idx].mean())
    return {
        "mean_difference": float(delta.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "block_hours": block,
        "replicates": repeats,
    }


def shortage(frame, start, hours=72, generation_factor=0.1):
    out = frame.copy()
    if not 0 <= generation_factor <= 1 or start < 0 or hours < 1:
        raise ValueError("Invalid stress window")
    out.loc[out.index[start : start + hours], ["pv_kw", "wind_kw"]] *= generation_factor
    return out
