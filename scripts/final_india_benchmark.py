"""Freeze validation-selected design, then evaluate the complete later test once.

This is a constructed reference case based on a secondary national-load mirror.
Its provenance limitations remain in the manifest; it is not a local field trial.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from microgrid.ageing import evaluate_lfp
from microgrid.calibration import by_source_horizon
from microgrid.controllers import deterministic_mpc, scenario_mpc
from microgrid.data import CHANNELS, audit, splits
from microgrid.evaluation import degradation, paired_block_bootstrap, reliability
from microgrid.feasibility import perfect_foresight_episode
from microgrid.forecasting import QuantileForecaster, residual_blocks
from microgrid.physics import Battery, BatteryConfig
from microgrid.reserve import vectorized_suffix_reserves
from microgrid.scenarios import sample_blocks


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def select_design(sizing_files):
    table = pd.concat([pd.read_csv(p) for p in sizing_files], ignore_index=True)
    eligible = table[
        (table.terminal == "initial") & (table.initial_soc == 0.5) & (table.lpsp_energy <= 0.01)
    ]
    if not len(eligible):
        raise ValueError("No design met predeclared validation adequacy gate")
    # Explicit non-economic lexicographic choice. No final test used here.
    return eligible.sort_values(["capacity_kwh", "renewable_scale", "power_kw"]).iloc[0].to_dict()


def trace_metrics(trace, c):
    result = reliability(trace, c.dt_hours)
    result["terminal_energy_kwh"] = float(trace.energy_kwh.iloc[-1])
    result["illustrative_ageing"] = degradation([c.initial_soc, *trace.soc], len(trace))
    published = evaluate_lfp([c.initial_soc, *trace.soc])
    published.pop("windows")
    result["published_model_ageing"] = published
    result["unserved_above_energy_floor_kwh"] = float(
        trace.loc[trace.energy_kwh > c.minimum + 1e-6, "unserved_kw"].sum()
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/india/reference/hourly.parquet")
    p.add_argument("--audit", default="data/india/reference/audit.json")
    p.add_argument(
        "--sizing",
        nargs="+",
        default=[
            "outputs/india_study/sizing/sizing.csv",
            "outputs/india_study/sizing_expanded/sizing.csv",
        ],
    )
    p.add_argument("--output", default="outputs/india_study/final")
    p.add_argument("--scenarios", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "frozen_manifest.json").exists():
        raise ValueError(
            "Frozen test run already exists; use another directory for an explicitly labelled replication"
        )
    frame = pd.read_parquet(args.data)
    audit(frame)
    design = select_design(args.sizing)
    c = BatteryConfig(capacity_kwh=design["capacity_kwh"], power_kw=design["power_kw"])
    frame[["pv_kw", "wind_kw"]] *= design["renewable_scale"]
    part = splits(len(frame))
    # Fitted only on outer training. Fixed fraction selected before outer calibration/test.
    model = QuantileForecaster(seed=args.seed, trees=60).fit(frame, part["train"].stop, stride=24)
    tuning_start = part["validation"].start
    tuning_stop = tuning_start + min(336, part["validation"].stop - tuning_start)
    tuning = frame.iloc[tuning_start:tuning_stop].reset_index(drop=True)
    forecasts = [model.predict(frame, t)[:, :, 1] for t in range(tuning_start, tuning_stop)]
    fixed_results = {}
    from microgrid.evaluation import dispatch

    for fraction in [0.1, 0.2, 0.4, 0.6, 0.8]:

        def action(t, e, x):
            forecast = forecasts[t].copy()
            forecast[0] = x
            return deterministic_mpc(forecast, e, c, reserve_kwh=fraction * c.capacity)

        _, result = dispatch(tuning, action, c)
        fixed_results[fraction] = result
    fixed = min(
        fixed_results,
        key=lambda f: (fixed_results[f]["eens_kwh"], fixed_results[f]["throughput_kwh"]),
    )
    residual = residual_blocks(model, frame, part["calibration"].start, part["calibration"].stop)
    model.save(output / "forecaster.pkl")
    np.save(output / "residual_blocks.npy", residual)
    methods = [
        "rule_based",
        "deterministic_mpc",
        "tuned_fixed_mpc",
        "adaptive_max_fallback",
        "adaptive_no_infeasible_target",
        "independent_reserve_mpc",
        "scenario_mpc",
    ]
    manifest = {
        "case": "constructed_Jaisalmer_weather_scaled_national_demand_proxy",
        "source_audit": json.loads(Path(args.audit).read_text()),
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((output / "forecaster.pkl").read_bytes()).hexdigest(),
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *Path("src/microgrid").glob("*.py")]
        },
        "battery": c.__dict__,
        "design_selection": "validation-only minimum capacity, then renewable scale, then power; not cost optimization",
        "selected_design": design,
        "fixed_fraction": fixed,
        "fixed_tuning_start": tuning_start,
        "fixed_tuning_stop": tuning_stop,
        "alpha": 0.05,
        "scenario_count": args.scenarios,
        "seed": args.seed,
        "residual_blocks": len(residual),
        "splits": {k: [s.start, s.stop] for k, s in part.items()},
        "methods": methods,
        "temperature_assumption_c": 25,
        "ageing_soh_feedback": False,
        "test_status": "configuration written before final dispatch metrics; no claim of formal preregistration",
    }
    dump(output / "selection.json", fixed_results)
    dump(output / "frozen_manifest.json", manifest)
    print(
        "FROZEN",
        json.dumps(
            {
                "battery": c.__dict__,
                "scale": design["renewable_scale"],
                "fixed": fixed,
                "residual_blocks": len(residual),
            }
        ),
        flush=True,
    )
    start, stop = part["test"].start, part["test"].stop
    batteries = {name: Battery(c) for name in methods}
    records = {name: [] for name in methods}
    timings = {name: [] for name in methods}
    shared = []
    reserves = []
    predictions = []
    truths = []
    for origin in range(start, stop):
        begin = time.perf_counter()
        q = model.predict(frame, origin)
        s = sample_blocks(q, residual, args.scenarios, args.seed + origin)
        independent = sample_blocks(
            q, residual, args.scenarios, args.seed + origin, independent=True
        )
        x = frame[CHANNELS].iloc[origin].to_numpy()
        point = q[:, :, 1].copy()
        s[:, 0], independent[:, 0], point[0] = x, x, x
        raw, flags, mass = vectorized_suffix_reserves(s, c)
        other, other_flags, _ = vectorized_suffix_reserves(independent, c)
        targets = np.where(flags[1:], c.maximum, raw[1:])
        relaxed = np.where(flags[1:], c.minimum, raw[1:])
        independent_targets = np.where(other_flags[1:], c.maximum, other[1:])
        prepare_seconds = time.perf_counter() - begin
        shared.append(prepare_seconds)
        reserves.append(
            {
                "timestamp": str(frame.timestamp.iloc[origin]),
                "initial_target_kwh": None if flags[0] else float(raw[0]),
                "initial_infeasible": bool(flags[0]),
                "initial_infeasible_mass": float(mass[0]),
                "end_target_kwh": None if flags[1] else float(raw[1]),
                "end_infeasible": bool(flags[1]),
            }
        )
        if origin + 24 <= stop and (origin - start) % 24 == 0:
            predictions.append(q)
            truths.append(frame[CHANNELS].iloc[origin : origin + 24].to_numpy())
        for name in methods:
            b = batteries[name]
            begin = time.perf_counter()
            if name == "rule_based":
                action = float(x[2] - x[0] - x[1])
            elif name == "deterministic_mpc":
                action = deterministic_mpc(point, b.energy, c)
            elif name == "tuned_fixed_mpc":
                action = deterministic_mpc(point, b.energy, c, reserve_kwh=fixed * c.capacity)
            elif name == "adaptive_max_fallback":
                action = deterministic_mpc(point, b.energy, c, reserve_kwh=targets)
            elif name == "adaptive_no_infeasible_target":
                action = deterministic_mpc(point, b.energy, c, reserve_kwh=relaxed)
            elif name == "independent_reserve_mpc":
                action = deterministic_mpc(point, b.energy, c, reserve_kwh=independent_targets)
            else:
                action = scenario_mpc(s, b.energy, c)
            duration = time.perf_counter() - begin
            r = b.step(action, *x)
            r["timestamp"] = frame.timestamp.iloc[origin]
            records[name].append(r)
            timings[name].append(duration)
        if (origin - start + 1) % 168 == 0:
            print(f"dispatched {origin - start + 1}/{stop - start} test hours", flush=True)
    result = {}
    traces = {}
    for name in methods:
        trace = pd.DataFrame(records[name])
        trace.to_parquet(output / f"trace_{name}.parquet", index=False)
        r = trace_metrics(trace, c)
        r["controller_seconds"] = sum(timings[name])
        r["controller_p95_ms"] = float(np.quantile(timings[name], 0.95) * 1000)
        r["shared_pipeline_seconds"] = sum(shared) if name != "rule_based" else 0
        r["runtime_scope"] = (
            "shared pipeline includes forecast, BOTH scenario samplers and reserve calculations; conservative common overhead"
        )
        result[name] = r
        traces[name] = trace
    oracle, ref = perfect_foresight_episode(frame.iloc[start:stop].reset_index(drop=True), c)
    oracle.to_parquet(output / "trace_perfect_foresight.parquet", index=False)
    result["perfect_foresight_non_deployable"] = {**trace_metrics(oracle, c), "oracle": ref}
    dump(output / "metrics.json", result)
    dump(output / "reserves.json", reserves)
    dump(
        output / "forecast_test_raw.json",
        by_source_horizon(np.array(truths), np.array(predictions)),
    )
    intervals = {}
    for block in [24, 168]:
        intervals[str(block)] = paired_block_bootstrap(
            traces["adaptive_max_fallback"].unserved_kw,
            traces["tuned_fixed_mpc"].unserved_kw,
            block=block,
            seed=args.seed,
        )
    dump(output / "paired_bootstrap.json", intervals)
    pd.DataFrame(
        {
            k: {m: v[m] for m in ["eens_kwh", "lpsp_energy", "curtailment_kwh", "throughput_kwh"]}
            for k, v in result.items()
        }
    ).T.to_csv(output / "comparison.csv")
    print(json.dumps({k: v["eens_kwh"] for k, v in result.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
