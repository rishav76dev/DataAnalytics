"""Primary-source audit and an explicitly constructed India reference dataset."""

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .data import audit, fit_capacity_scaling, pv_power, wind_power

DOI = "10.17632/y58jknpgs8.2"


def primary_demand(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def get(url):
        with urlopen(
            Request(
                url, headers={"User-Agent": "adaptive-microgrid-research/0.2", "Accept": "*/*"}
            ),
            timeout=60,
        ) as r:
            return r.read()

    metadata = json.loads(get(f"https://api.datacite.org/dois/{DOI}"))
    entries = json.loads(
        get(
            "https://data.mendeley.com/public-api/datasets/y58jknpgs8/files?folder_id=root&version=2"
        )
    )
    rights = metadata["data"]["attributes"]["rightsList"]
    if not any(x.get("rightsIdentifier") == "cc-by-4.0" for x in rights):
        raise ValueError("Primary data licence not confirmed as expected")
    entry = next(e for e in entries if e["filename"] == "January 2024- June 2025.xlsx")
    dest = output / "2024_2025.xlsx"
    if not dest.exists():
        dest.write_bytes(get(entry["content_details"]["download_url"]))
    checksum = hashlib.sha256(dest.read_bytes()).hexdigest()
    if checksum != entry["content_details"]["sha256_hash"]:
        raise ValueError("Primary workbook checksum mismatch")
    raw = pd.read_excel(dest, sheet_name="Report")
    stamp = pd.to_datetime(raw["Timestamp"], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    f = pd.DataFrame(
        {
            "timestamp": stamp,
            "demand_mw": pd.to_numeric(raw["Demand (MW)"], errors="coerce"),
            "solar_mw": pd.to_numeric(raw["Solar (MW)"], errors="coerce"),
        }
    )
    f = f.loc[stamp.notna()].reset_index(drop=True)
    if (
        f.timestamp.duplicated().any()
        or not f.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    ):
        raise ValueError("Primary hourly series is not continuous")
    if not np.isfinite(f.demand_mw).all() or (f.demand_mw <= 0).any():
        raise ValueError("Primary demand has invalid records")
    (output / "datacite.json").write_text(json.dumps(metadata, indent=2))
    (output / "files_manifest.json").write_text(json.dumps(entries, indent=2))
    f.to_parquet(output / "demand.parquet", index=False)
    return f, {
        "doi": DOI,
        "primary_sha256_verified": checksum,
        "primary_rows": len(f),
        "primary_removed_summary_rows": int(stamp.isna().sum()),
        "licence": "CC BY 4.0, verified via DataCite rights metadata",
        "creators": metadata["data"]["attributes"]["creators"],
        "title": metadata["data"]["attributes"]["titles"],
        "units": "MW in primary workbook column header",
        "timezone": "Naive source timestamps interpreted as Asia/Kolkata; contextual assumption, not explicit in registry",
        "solar_peak_hour_assumption_check": int(
            f.groupby(f.timestamp.dt.hour).solar_mw.mean().idxmax()
        ),
    }


def build_reference(weather_paths, mirror_path, primary_dir, output):
    primary, provenance = primary_demand(primary_dir)
    if (
        hashlib.sha256(Path(mirror_path).read_bytes()).hexdigest()
        != "c77ed3e200565c5609bd52806bef98f4e7eac22da6609d60750c8ce7a5fb75ed"
    ):
        raise ValueError("Pinned mirror checksum mismatch")
    mirror = pd.read_parquet(mirror_path)
    common = mirror.index.intersection(pd.DatetimeIndex(primary.timestamp))
    primary_values = primary.set_index("timestamp").loc[common].demand_mw.to_numpy()
    difference = float(np.abs(mirror.loc[common].demand.to_numpy() - primary_values).max())
    if difference > 1e-6:
        raise ValueError("Primary demand does not agree with mirror")
    # 2024/25 use publisher values directly. 2022/23 remain the pinned hourly mirror.
    demand = mirror.demand.copy()
    demand.loc[common] = primary_values
    demand = demand.loc["2022-01-01":"2025-06-24 23:00:00"]
    bad = (demand < 50000) | (demand > 300000) | ~np.isfinite(demand)
    demand = demand.mask(bad).ffill(limit=3)
    if demand.isna().any():
        raise ValueError("Unresolved demand gap")
    local = pd.DatetimeIndex(demand.index).tz_localize("Asia/Kolkata")
    utc = local.tz_convert("UTC")
    w = pd.concat([pd.read_parquet(p) for p in weather_paths]).set_index("timestamp").sort_index()
    if w.index.duplicated().any():
        raise ValueError("Overlapping weather sources")
    floor = utc.floor("h")
    a, b = w.reindex(floor), w.reindex(floor + pd.Timedelta(hours=1))
    if a.isna().any().any() or b.isna().any().any():
        raise ValueError("Incomplete overlapping weather hours")
    frame = pd.DataFrame((a.to_numpy() + b.to_numpy()) / 2, columns=w.columns)
    frame.insert(0, "timestamp", utc)
    frame["load_kw"] = demand.to_numpy() * 1000
    train = local.year == 2022
    scale = 12 / frame.loc[train, "load_kw"].mean()
    frame["load_kw"] *= scale
    capacities = fit_capacity_scaling(frame.loc[train])
    frame["pv_kw"] = (
        pv_power(frame.irradiance_wm2, frame.temperature_c, capacities["pv_capacity_kw"]) * 2
    )
    frame["wind_kw"] = wind_power(frame.wind_speed_ms, capacities["wind_capacity_kw"]) * 2
    report = audit(frame)
    report.update(
        {
            "source": provenance,
            "mirror_primary_max_absolute_difference_mw": difference,
            "mirror_primary_compared_hours": len(common),
            "national_profile_not_local_site": True,
            "mirror_path_sha256": hashlib.sha256(Path(mirror_path).read_bytes()).hexdigest(),
            "pre_2024_data": "pinned hourly mirror; full raw pre-2024 reaggregation not independently repeated",
            "repair_count": int(bad.sum()),
            "repair": "causal forward-fill of <=3 consecutive implausible demand hours",
            "demand_training_year": 2022,
            "demand_scale": float(scale),
            "generation_reference_capacities": capacities,
            "renewable_multiplier": 2,
            "weather_alignment": "piecewise-constant overlap averaging across UTC/IST half-hour offset",
            "causality_scope": "Forecast causal on processed power series; resampling assumptions are not field measurement availability validation",
            "wind_model": "simplified 10m wind power curve; no independent turbine or pack site validation",
            "partitions": {
                "fit": "2022",
                "residual_pool": "2023",
                "tail_calibration": "2024-01 through 2024-06",
                "selection": "2024-07 through 2024-12 (development, previously inspected)",
                "fresh_test": "2025-01-01 through 2025-06-24",
            },
        }
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output / "hourly.parquet", index=False)
    (output / "audit.json").write_text(json.dumps(report, indent=2))
    return frame, report
