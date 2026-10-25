"""Strict India weather ingestion. No synthetic substitution for missing demand."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .data import audit, fit_capacity_scaling, pv_power, splits, wind_power

PARAMETERS = {
    "ALLSKY_SFC_SW_DWN": "irradiance_wm2",
    "T2M": "temperature_c",
    "WS10M": "wind_speed_ms",
}


def parse_power(payload):
    if payload.get("header", {}).get("time_standard", "").upper() != "UTC":
        raise ValueError("NASA response must explicitly declare UTC, not local solar time")
    expected_units = {
        "ALLSKY_SFC_SW_DWN": {"Wh/m^2", "W-hr/m^2", "W/m^2"},
        "T2M": {"C"},
        "WS10M": {"m/s"},
    }
    for key, units in expected_units.items():
        if payload["parameters"][key]["units"] not in units:
            raise ValueError(f"Unexpected units for {key}; inspect API metadata")
    values = payload["properties"]["parameter"]
    if any(set(values[k]) != set(values["T2M"]) for k in PARAMETERS):
        raise ValueError("Weather parameters have mismatched timestamps")
    frame = pd.DataFrame({name: values[k] for k, name in PARAMETERS.items()}).sort_index()
    frame.insert(0, "timestamp", pd.to_datetime(frame.index, format="%Y%m%d%H", utc=True))
    frame = frame.reset_index(drop=True)
    a = frame[list(PARAMETERS.values())].to_numpy(dtype=float)
    fill = payload.get("header", {}).get("fill_value", -999)
    if not np.isfinite(a).all() or (a == fill).any():
        raise ValueError("Missing weather values require an explicit repair policy")
    if (frame.irradiance_wm2 < 0).any() or (frame.wind_speed_ms < 0).any():
        raise ValueError("Negative irradiance or wind")
    if len(frame) < 24 or not frame.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Weather must be continuous hourly")
    # Hourly Wh/m² divided by exactly one hour equals mean W/m² numerically.
    return frame


def fetch_weather(output, start_year=2020, end_year=2024):
    if not 1984 <= start_year <= end_year < datetime.now(timezone.utc).year:
        raise ValueError("Choose complete historical years from 1984 onward")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames, sources = [], []
    for year in range(start_year, end_year + 1):
        params = dict(
            parameters=",".join(PARAMETERS),
            community="RE",
            longitude=70.9083,
            latitude=26.9157,
            start=f"{year}0101",
            end=f"{year}1231",
            format="JSON",
            **{"time-standard": "UTC"},
        )
        url = "https://power.larc.nasa.gov/api/temporal/hourly/point?" + urlencode(params)
        raw = output / f"power_{year}.json"
        if raw.exists():
            body = raw.read_bytes()
        else:
            with urlopen(url, timeout=60) as response:
                body = response.read()
            # Validate response before preserving it as usable raw input.
            parse_power(json.loads(body))
            raw.write_bytes(body)
        f = parse_power(json.loads(body))
        expected = pd.date_range(
            f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left", tz="UTC"
        )
        if not pd.DatetimeIndex(f.timestamp).equals(expected):
            raise ValueError(f"Incomplete weather year {year}")
        frames.append(f)
        sources.append({"url": url, "file": raw.name, "sha256": hashlib.sha256(body).hexdigest()})
    weather = pd.concat(frames, ignore_index=True)
    weather.to_parquet(output / "weather.parquet", index=False)
    manifest = {
        "site": "Jaisalmer, Rajasthan, India",
        "latitude": 26.9157,
        "longitude": 70.9083,
        "timezone": "UTC",
        "sources": sources,
        "retrieved_or_cache_checked_utc": datetime.now(timezone.utc).isoformat(),
        "data_kind": "NASA POWER weather estimates, not measured plant generation",
        "licence_status": "Verify and archive current NASA terms before redistribution",
        "wind_height_m": 10,
        "irradiance_conversion": "hourly Wh/m2 / 1h -> W/m2",
    }
    (output / "weather_manifest.json").write_text(json.dumps(manifest, indent=2))
    return weather


def prepare_india(weather_path, load_path, metadata_path, output):
    """Require audited demand metadata; exact timestamp joins, no silent gap filling."""
    metadata = json.loads(Path(metadata_path).read_text())
    required = [
        "source_url",
        "licence",
        "site",
        "timezone",
        "interval_convention",
        "load_kind",
        "wind_height_assumption",
        "weather_terms_reference",
    ]
    if any(
        not isinstance(metadata.get(k), str) or not metadata[k].strip() or "REPLACE_" in metadata[k]
        for k in required
    ):
        raise ValueError(f"Demand/source metadata requires: {required}")
    if metadata.get("units") != "kW" or metadata.get("reuse_permitted") is not True:
        raise ValueError("Confirm demand units kW and documented permission to reuse")
    if metadata["load_kind"] not in {"measured_local", "scaled_aggregate_proxy"}:
        raise ValueError(
            "Declare measured_local or scaled_aggregate_proxy, never hide synthetic load"
        )
    if metadata["interval_convention"] != "start":
        raise ValueError("Convert interval-end readings to interval-start before this importer")
    weather = pd.read_parquet(weather_path)
    load = pd.read_csv(load_path)
    if set(["timestamp", "load_kw"]) - set(load):
        raise ValueError("Demand CSV requires timestamp and load_kw")
    if any(pd.Timestamp(t).tzinfo is None for t in load.timestamp):
        raise ValueError("Demand timestamps must have explicit timezone offsets")
    load["timestamp"] = pd.to_datetime(load.timestamp, utc=True)
    weather["timestamp"] = pd.to_datetime(weather.timestamp, utc=True)
    if load.timestamp.duplicated().any() or weather.timestamp.duplicated().any():
        raise ValueError("Duplicate source timestamps")
    start, stop = (
        max(load.timestamp.min(), weather.timestamp.min()),
        min(load.timestamp.max(), weather.timestamp.max()),
    )
    weather = weather[weather.timestamp.between(start, stop)]
    demand = load[load.timestamp.between(start, stop)]
    if not len(weather) or set(weather.timestamp) != set(demand.timestamp):
        raise ValueError(
            "Demand/weather hourly windows do not align; do not round IST offsets or drop gaps"
        )
    frame = (
        weather.merge(demand[["timestamp", "load_kw"]], on="timestamp", validate="one_to_one")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    vals = frame[["irradiance_wm2", "temperature_c", "wind_speed_ms", "load_kw"]].to_numpy(float)
    if (
        not np.isfinite(vals).all()
        or (frame[["irradiance_wm2", "wind_speed_ms", "load_kw"]] < 0).any().any()
    ):
        raise ValueError("Invalid weather/demand values")
    capacity = fit_capacity_scaling(frame.iloc[splits(len(frame))["train"]])
    frame["pv_kw"] = pv_power(frame.irradiance_wm2, frame.temperature_c, capacity["pv_capacity_kw"])
    frame["wind_kw"] = wind_power(frame.wind_speed_ms, capacity["wind_capacity_kw"])
    report = audit(frame)
    report.update(
        {
            "source_metadata": metadata,
            "capacity_fit": "training only",
            "capacities": capacity,
            "weather_sha256": hashlib.sha256(Path(weather_path).read_bytes()).hexdigest(),
            "load_sha256": hashlib.sha256(Path(load_path).read_bytes()).hexdigest(),
            "generation_model": "simplified engineering models; 10m wind input, not validated hub-height output",
            "case_status": "constructed Indian reference microgrid, not measured co-located generation",
        }
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "hourly.parquet", index=False)
    frame.to_csv(output / "hourly.csv", index=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2))
    return frame
