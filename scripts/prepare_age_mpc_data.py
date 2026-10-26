"""Fetch and verify inputs for the constructed India degradation-MPC study."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

from prepare_india_reference import EXPECTED_SHA256, URL

from microgrid.india import fetch_weather
from microgrid.research_data import build_reference

if __name__ == "__main__":
    output = Path("data/india/age_mpc_v2")
    if (output / "hourly.parquet").exists():
        raise ValueError("Prepared inputs already exist; preserve frozen data")
    for directory, first, last in [("weather", 2020, 2024), ("weather_2025", 2025, 2025)]:
        fetch_weather(Path("data/india") / directory, first, last)
    mirror = Path("data/india/demand/hourly_raw.parquet")
    if not mirror.exists():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(URL, timeout=60) as response:
            body = response.read()
        if hashlib.sha256(body).hexdigest() != EXPECTED_SHA256:
            raise ValueError("Downloaded mirror checksum mismatch")
        mirror.write_bytes(body)
    frame, audit = build_reference(
        ["data/india/weather/weather.parquet", "data/india/weather_2025/weather.parquet"],
        mirror,
        "data/india/primary",
        output,
    )
    print(f"Verified {len(frame)} hours; primary overlap {audit['mirror_primary_compared_hours']}")
