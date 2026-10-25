"""Construct a reference case; national demand proxy is NOT a local measured load."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from microgrid.data import audit, fit_capacity_scaling, pv_power, splits, wind_power

EXPECTED_SHA256 = "c77ed3e200565c5609bd52806bef98f4e7eac22da6609d60750c8ce7a5fb75ed"
REPO_COMMIT = "9fc756aeceefb2e53d6494d48722f5dd771996e0"
URL = f"https://raw.githubusercontent.com/RudranshVyas/india-grid-forecast/{REPO_COMMIT}/data/processed/hourly_raw.parquet"


def prepare(weather, demand, output, demand_mean_kw=12.0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    w = pd.read_parquet(weather).set_index("timestamp")
    if hashlib.sha256(Path(demand).read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Pinned demand mirror checksum mismatch")
    raw = pd.read_parquet(demand)
    # Explicit documented interpretation, based on the mirror data dictionary.
    # The original telemetry timezone has not been independently author-confirmed.
    local = pd.DatetimeIndex(raw.index).tz_localize("Asia/Kolkata")
    demand_mw = pd.Series(raw.demand.to_numpy(), index=local).loc[
        "2022-01-01":"2024-12-31 23:00:00"
    ]
    bad = (demand_mw < 50000) | (demand_mw > 300000) | ~np.isfinite(demand_mw)
    cleaned = demand_mw.mask(bad).ffill(limit=3)
    if cleaned.isna().any():
        raise ValueError("Long missing demand run: do not silently repair")
    ts = cleaned.index.tz_convert("UTC")
    # IST hour-start intervals overlap two UTC weather hours equally.
    floor = ts.floor("h")
    left = w.reindex(floor)
    right = w.reindex(floor + pd.Timedelta(hours=1))
    if left.isna().any().any() or right.isna().any().any():
        raise ValueError("Weather does not cover both half-hour overlap bins")
    weather_values = 0.5 * (left.to_numpy() + right.to_numpy())
    frame = pd.DataFrame(weather_values, columns=w.columns)
    frame.insert(0, "timestamp", ts)
    frame["load_kw"] = cleaned.to_numpy() * 1000
    training = splits(len(frame))["train"]
    scale = demand_mean_kw / frame.load_kw.iloc[training].mean()
    frame["load_kw"] *= scale
    caps = fit_capacity_scaling(frame.iloc[training])
    frame["pv_kw"] = pv_power(frame.irradiance_wm2, frame.temperature_c, caps["pv_capacity_kw"])
    frame["wind_kw"] = wind_power(frame.wind_speed_ms, caps["wind_capacity_kw"])
    report = audit(frame)
    report.update(
        {
            "case": "constructed_indian_reference_not_measured_local_microgrid",
            "weather_source": "NASA POWER Jaisalmer gridded weather",
            "demand_source_doi": "10.17632/y58jknpgs8.2",
            "demand_authors": "Kumar, S.; Mukherjee, D.; Kalita, K. (2025)",
            "source_url": URL,
            "mirror_commit": REPO_COMMIT,
            "source_raw_sha256": hashlib.sha256(Path(demand).read_bytes()).hexdigest(),
            "weather_sha256": hashlib.sha256(Path(weather).read_bytes()).hexdigest(),
            "licence": "CC BY 4.0 reported in mirror documentation; primary Mendeley record blocked by proxy",
            "primary_source_verification": "pending_not_independently_verified",
            "demand_kind": "scaled_national_aggregate_proxy",
            "units_before_scaling": "MW, according to mirror data dictionary",
            "timezone_interpretation": "Asia/Kolkata according to mirror; not independently source-confirmed",
            "weather_alignment": "half-overlap mean of adjacent UTC hours, piecewise constant hourly weather assumption",
            "demand_outliers_repaired": int(bad.sum()),
            "repair": "causal forward fill, maximum 3 hours, bounds 50,000–300,000 MW",
            "demand_target_training_mean_kw": demand_mean_kw,
            "demand_scale": float(scale),
            "generation_capacities_training_only": caps,
            "generation_model": "simplified PV derate and wind curve at 10m; no hub-height validation",
            "split_indices": {k: [s.start, s.stop] for k, s in splits(len(frame)).items()},
        }
    )
    frame.to_parquet(output / "hourly.parquet", index=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2))
    return frame, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weather", default="data/india/weather/weather.parquet")
    p.add_argument("--demand", default="data/india/demand/hourly_raw.parquet")
    p.add_argument("--output", default="data/india/reference")
    args = p.parse_args()
    dest = Path(args.demand)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=60) as response:
            dest.write_bytes(response.read())
    frame, report = prepare(args.weather, args.demand, args.output)
    print(
        json.dumps(
            {
                "rows": len(frame),
                "start": report["start"],
                "end": report["end"],
                "repaired": report["demand_outliers_repaired"],
                "capacities": report["generation_capacities_training_only"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
