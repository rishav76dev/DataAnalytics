import json
from pathlib import Path

import numpy as np
import pandas as pd

CHANNELS = ["pv_kw", "wind_kw", "load_kw"]


def synthetic(hours=2160, seed=42):
    """Demonstration data only; no empirical or public-dataset claim."""
    if hours < 24:
        raise ValueError("At least 24 hours required")
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    cloud = np.zeros(hours)
    wind = np.zeros(hours)
    for i in range(1, hours):
        cloud[i] = 0.9 * cloud[i - 1] + rng.normal(0, 0.15)
        wind[i] = 0.95 * wind[i - 1] + rng.normal(0, 0.5)
    irradiance = 1000 * np.maximum(np.sin((t % 24 - 6) * np.pi / 12), 0)
    irradiance *= np.clip(0.8 + cloud, 0.1, 1)
    speed = np.clip(7 + wind - cloud, 0, 22)
    load = np.maximum(12 + 5 * np.sin((t % 24 - 14) * np.pi / 12) + rng.normal(0, 1, hours), 1)
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC"),
            "irradiance_wm2": irradiance,
            "wind_speed_ms": speed,
            "temperature_c": 25 + 6 * np.sin(t * np.pi / 12),
            "load_kw": load,
        }
    )
    frame["pv_kw"] = pv_power(irradiance, frame.temperature_c, 45)
    frame["wind_kw"] = wind_power(speed, 20)
    return frame


def pv_power(irradiance_wm2, temperature_c, capacity_kw):
    if capacity_kw <= 0:
        raise ValueError("Capacity must be positive")
    # Simplified DC temperature correction and fixed inverter/system derate.
    return np.clip(
        capacity_kw
        * np.asarray(irradiance_wm2)
        / 1000
        * (1 - 0.004 * (np.asarray(temperature_c) - 25))
        * 0.9,
        0,
        capacity_kw,
    )


def wind_power(speed_ms, capacity_kw):
    if capacity_kw <= 0:
        raise ValueError("Capacity must be positive")
    v = np.asarray(speed_ms)
    return capacity_kw * np.where(
        (v < 3) | (v >= 25), 0, np.clip((v**3 - 3**3) / (12**3 - 3**3), 0, 1)
    )


def audit(frame):
    missing = set(["timestamp", *CHANNELS]) - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    ts = pd.to_datetime(frame.timestamp, utc=True, errors="raise")
    if len(frame) < 24 or ts.isna().any() or ts.duplicated().any():
        raise ValueError("Need at least 24 unique, valid hourly timestamps")
    if not ts.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError(
            "Timestamps must be ordered and continuous hourly; gaps require explicit repair"
        )
    values = frame[CHANNELS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Power must be finite, nonnegative kW")
    return {
        "rows": len(frame),
        "start": str(ts.iloc[0]),
        "end": str(ts.iloc[-1]),
        "missing_hours": 0,
        "units": {c: "kW" for c in CHANNELS},
        "energy_kwh": frame[CHANNELS].sum().to_dict(),
    }


def splits(length, fractions=(0.60, 0.15, 0.10, 0.15)):
    if (
        length < 4
        or len(fractions) != 4
        or min(fractions) <= 0
        or not np.isclose(sum(fractions), 1)
    ):
        raise ValueError("Need four positive fractions summing to one")
    cuts = [0, *np.cumsum([int(length * x) for x in fractions[:3]]).tolist(), length]
    if any(b <= a for a, b in zip(cuts[:-1], cuts[1:])):
        raise ValueError("Split too small")
    return {
        name: slice(a, b)
        for name, a, b in zip(["train", "validation", "calibration", "test"], cuts[:-1], cuts[1:])
    }


def fit_capacity_scaling(training, pv_share=0.7, energy_margin=1.15):
    """Return capacities from training-only unit-capacity weather conversion."""
    load = training.load_kw.mean()
    pv = pv_power(training.irradiance_wm2, training.temperature_c, 1).mean()
    wind = wind_power(training.wind_speed_ms, 1).mean()
    if pv <= 0 or wind <= 0 or not 0 <= pv_share <= 1 or energy_margin <= 0:
        raise ValueError("Invalid scaling inputs")
    return {
        "pv_capacity_kw": float(load * energy_margin * pv_share / pv),
        "wind_capacity_kw": float(load * energy_margin * (1 - pv_share) / wind),
    }


def prepare(output, csv=None, hours=2160, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(csv) if csv else synthetic(hours, seed)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    report = audit(frame)
    report["provenance"] = str(csv) if csv else "synthetic demonstration; not research evidence"
    report["license"] = "User must audit source licence" if csv else "Generated locally"
    report["splits"] = {k: [v.start, v.stop] for k, v in splits(len(frame)).items()}
    frame.to_parquet(output / "hourly.parquet", index=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2))
    return frame
