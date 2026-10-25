"""India acquisition and validation CLI. No automatic final-test evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from microgrid.calibration import IntervalCalibrator, by_source_horizon
from microgrid.data import CHANNELS, audit, splits
from microgrid.feasibility import sizing_study
from microgrid.forecasting import QuantileForecaster, residual_blocks
from microgrid.india import fetch_weather, prepare_india


def calibration_check(frame, output, seed=42):
    parts = splits(len(frame))
    model = QuantileForecaster(seed=seed).fit(frame, parts["train"].stop, stride=24)
    midpoint = (parts["validation"].start + parts["validation"].stop) // 2

    def samples(start, stop):
        origins = list(range(start, stop - 23, 24))
        if not origins:
            raise ValueError("Insufficient history for calibration evaluation")
        return (
            origins,
            np.array([frame[CHANNELS].iloc[t : t + 24].to_numpy() for t in origins]),
            np.array([model.predict(frame, t) for t in origins]),
        )

    _, yfit, qfit = samples(parts["validation"].start, midpoint)
    calibrator = IntervalCalibrator().fit(yfit, qfit, midpoint)
    origins, truth, raw = samples(midpoint, parts["validation"].stop)
    corrected = np.array([calibrator.apply(q, t) for q, t in zip(raw, origins)])
    # Separate later outer calibration window for scenario residual trajectories.
    residual = residual_blocks(model, frame, parts["calibration"].start, parts["calibration"].stop)
    report = {
        "status": "held_out_validation_diagnostic_not_final_test",
        "training_stop": parts["train"].stop,
        "correction_fit_end": midpoint,
        "correction_fit_origins": len(yfit),
        "evaluation_origins": len(origins),
        "raw": by_source_horizon(truth, raw),
        "corrected": by_source_horizon(truth, corrected),
        "residual_blocks": len(residual),
        "scenario_note": "CQR widening does not modify or calibrate P50 residual bootstrap tails",
        "assumptions": "Temporal dependence: no exchangeable coverage guarantee asserted",
    }
    np.save(output / "interval_correction.npy", calibrator.correction)
    np.save(output / "residual_blocks.npy", residual)
    model.save(output / "forecaster.pkl")
    (output / "calibration_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch-weather")
    p.add_argument("--start-year", type=int, default=2020)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--output", default="data/india/weather")
    p = sub.add_parser("prepare")
    p.add_argument("--weather", required=True)
    p.add_argument("--load", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--output", default="data/india/processed")
    for command in ["feasibility", "calibration", "reserve"]:
        p = sub.add_parser(command)
        p.add_argument("--data", required=True)
        p.add_argument(
            "--data-status",
            required=True,
            choices=["synthetic", "constructed_india", "measured_india"],
        )
        p.add_argument("--output", required=True)
        if command == "reserve":
            p.add_argument("--hours", type=int, default=72)
        if command == "feasibility":
            p.add_argument("--capacities", type=float, nargs="+", default=[100, 200, 400])
            p.add_argument("--powers", type=float, nargs="+", default=[25, 50, 100])
            p.add_argument(
                "--renewable-scales", type=float, nargs="+", default=[0.75, 1, 1.25, 1.5]
            )
            p.add_argument("--initial-socs", type=float, nargs="+", default=[0.5])
            p.add_argument(
                "--terminals", nargs="+", choices=["minimum", "initial"], default=["minimum"]
            )
    args = parser.parse_args()
    if args.command == "fetch-weather":
        fetch_weather(args.output, args.start_year, args.end_year)
        return
    if args.command == "prepare":
        prepare_india(args.weather, args.load, args.metadata, args.output)
        return
    path, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    audit(frame)
    parts = splits(len(frame))
    manifest = {
        "data_status_user_declared": args.data_status,
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "partition": "validation_only",
        "final_test_evaluated": False,
        "arguments": vars(args),
        "splits": {k: [v.start, v.stop] for k, v in parts.items()},
    }
    # Written before collecting outcomes. This is not a claim of preregistration.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.command == "feasibility":
        selected = frame.iloc[parts["validation"]].reset_index(drop=True)
        result = sizing_study(
            selected,
            args.capacities,
            args.powers,
            args.renewable_scales,
            args.initial_socs,
            args.terminals,
        )
        result.to_csv(output / "sizing.csv", index=False)
        print(
            result[
                [
                    "capacity_kwh",
                    "power_kw",
                    "renewable_scale",
                    "initial_soc",
                    "terminal",
                    "lpsp_energy",
                ]
            ].to_string(index=False)
        )
    elif args.command == "reserve":
        from microgrid.reserve_study import validate_reserve

        result = validate_reserve(frame, output, hours=args.hours)
        print(
            json.dumps(
                {
                    "fixed_fraction": result["chosen_fixed_fraction"],
                    "eens_kwh": {k: v["eens_kwh"] for k, v in result["results"].items()},
                },
                indent=2,
            )
        )
    else:
        result = calibration_check(frame, output)
        print(
            json.dumps(
                {
                    "raw_coverage": result["raw"]["pooled"]["coverage_80"],
                    "corrected_coverage": result["corrected"]["pooled"]["coverage_80"],
                    "historical_residual_blocks": result["residual_blocks"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
