import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .controllers import deterministic_mpc, rule_based, scenario_mpc
from .data import CHANNELS, audit, prepare, splits
from .evaluation import dispatch, paired_block_bootstrap, shortage
from .forecasting import QuantileForecaster, metrics, residual_blocks
from .physics import BatteryConfig
from .reserve import ReserveEstimator, end_of_step_targets
from .scenarios import sample_blocks


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def load_config(path):
    if path is None:
        return BatteryConfig()
    return BatteryConfig(**yaml.safe_load(Path(path).read_text())["battery"])


def demo(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    c = load_config(args.config)
    frame = prepare(output / "data", args.csv, args.hours, args.seed)
    split = splits(len(frame))
    if split["train"].stop < 220 or split["calibration"].stop - split["calibration"].start < 48:
        raise ValueError("Need more history (use at least 720 hours)")
    model = QuantileForecaster(seed=args.seed, trees=args.trees).fit(frame, split["train"].stop)
    model.save(output / "forecaster.pkl")
    residual = residual_blocks(model, frame, split["calibration"].start, split["calibration"].stop)
    np.save(output / "calibration_residuals.npy", residual)
    origins = list(range(split["validation"].start, split["validation"].stop - 23, 24))
    pred = np.array([model.predict(frame, t) for t in origins])
    truth = np.array([frame[CHANNELS].iloc[t : t + 24].to_numpy() for t in origins])
    save_json(output / "forecast_validation.json", metrics(truth, pred))
    start = split["test"].start
    count = min(args.dispatch_hours, len(frame) - start)
    if count < 24:
        raise ValueError("Need at least 24 evaluation hours")
    # Freeze design + exact inputs BEFORE reading final-test metrics.
    manifest = {
        "seed": args.seed,
        "data_sha256": hashlib.sha256((output / "data/hourly.parquet").read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((output / "forecaster.pkl").read_bytes()).hexdigest(),
        "battery": c.__dict__,
        "evaluation_start": start,
        "evaluation_hours": count,
        "scenario_count": args.scenarios,
        "alpha": args.alpha,
        "controllers": [
            "rule_based",
            "deterministic_mpc",
            "fixed_reserve_mpc",
            "adaptive_reserve_mpc",
            "independent_reserve_mpc",
            "scenario_mpc",
            "perfect_foresight_non_deployable",
        ],
        "data_status": "user_supplied_unverified_licence" if args.csv else "synthetic_demo",
        "ageing_status": "illustrative_unvalidated",
        "design_status": "engineering_demo_not_preregistered_study",
    }
    save_json(output / "frozen_manifest.json", manifest)
    forecasts, joint, reserve_rows, joint_targets, independent_targets = [], [], [], [], []
    timing = []
    estimator = ReserveEstimator(c, args.alpha)
    # All uncertainty is calibrated before the first evaluation origin.
    for t in range(start, start + count):
        begin = time.perf_counter()
        q = model.predict(frame, t)
        measured = frame[CHANNELS].iloc[t].to_numpy()
        j = sample_blocks(q, residual, args.scenarios, args.seed + t)
        i = sample_blocks(q, residual, args.scenarios, args.seed + t, independent=True)
        f = q[:, :, 1].copy()
        # Current measured bus values are identical across scenarios at dispatch.
        j[:, 0], i[:, 0], f[0] = measured, measured, measured
        r, ri = estimator.estimate(j), estimator.estimate(i)
        forecasts.append(f)
        joint.append(j)
        jt, infeasible = end_of_step_targets(j, estimator)
        it, _ = end_of_step_targets(i, estimator)
        joint_targets.append(jt)
        independent_targets.append(it)
        reserve_rows.append(
            {
                "timestamp": str(frame.timestamp.iloc[t]),
                **r,
                "independent_target_kwh": ri["target_kwh"],
                "independent_risk_feasible": ri["risk_feasible"],
                "end_of_step_targets_kwh": jt.tolist(),
                "end_of_step_infeasible": infeasible,
            }
        )
        timing.append(time.perf_counter() - begin)
    save_json(output / "reserves.json", reserve_rows)
    evaluated = frame.iloc[start : start + count].reset_index(drop=True)

    controllers = {
        "rule_based": lambda t, e, x: rule_based(x),
        "deterministic_mpc": lambda t, e, x: deterministic_mpc(forecasts[t], e, c),
        "fixed_reserve_mpc": lambda t, e, x: deterministic_mpc(
            forecasts[t], e, c, reserve_kwh=0.4 * c.capacity
        ),
        # Infeasible reserves are logged; fallback is the physical maximum as a soft target.
        "adaptive_reserve_mpc": lambda t, e, x: deterministic_mpc(
            forecasts[t], e, c, reserve_kwh=joint_targets[t]
        ),
        "independent_reserve_mpc": lambda t, e, x: deterministic_mpc(
            forecasts[t],
            e,
            c,
            reserve_kwh=independent_targets[t],
        ),
        "scenario_mpc": lambda t, e, x: scenario_mpc(joint[t], e, c),
        "perfect_foresight_non_deployable": lambda t, e, x: deterministic_mpc(
            evaluated[CHANNELS].iloc[t:].to_numpy(), e, c
        ),
    }
    results, traces = {}, {}
    for name, controller in controllers.items():
        trace, result = dispatch(evaluated, controller, c)
        trace.to_parquet(output / f"trace_{name}.parquet", index=False)
        result["shared_forecast_scenario_reserve_seconds"] = (
            sum(timing) if name not in {"rule_based", "perfect_foresight_non_deployable"} else 0
        )
        # Shared pipeline timing is deliberately conservative for point-forecast methods.
        results[name], traces[name] = result, trace
        print(
            f"{name}: EENS={result['eens_kwh']:.3f} kWh, LPSP={result['lpsp_energy']:.4f}",
            flush=True,
        )
    save_json(output / "metrics.json", results)
    comparison = paired_block_bootstrap(
        traces["adaptive_reserve_mpc"].unserved_kw,
        traces["fixed_reserve_mpc"].unserved_kw,
        seed=args.seed,
    )
    save_json(output / "paired_bootstrap.json", comparison)
    # A separate synthetic stress diagnostic, never mixed into primary metrics.
    stressed = shortage(evaluated, 0, min(72, count))
    _, stress = dispatch(stressed, lambda t, e, x: rule_based(x), c)
    save_json(output / "shortage_rule_based.json", stress)
    table = pd.DataFrame(
        {
            k: {
                m: v[m]
                for m in [
                    "eens_kwh",
                    "lpsp_energy",
                    "curtailment_kwh",
                    "throughput_kwh",
                    "controller_seconds",
                ]
            }
            for k, v in results.items()
        }
    ).T
    table.to_csv(output / "comparison.csv")
    plot_results(output, traces, reserve_rows)
    print(f"Artifacts saved to {output.resolve()}")


def plot_results(output, traces, reserves):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/microgrid-mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/microgrid-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    first = next(iter(traces.values()))
    axes[0].plot(first.timestamp, first.pv_kw + first.wind_kw, label="Renewables")
    axes[0].plot(first.timestamp, first.load_kw, label="Demand")
    for name in ["rule_based", "fixed_reserve_mpc", "adaptive_reserve_mpc"]:
        x = traces[name]
        axes[1].plot(x.timestamp, x.energy_kwh, label=name)
        axes[2].plot(x.timestamp, x.unserved_kw, label=name)
    axes[1].plot(
        first.timestamp,
        [r["target_kwh"] if r["target_kwh"] is not None else np.nan for r in reserves],
        "k--",
        label="Adaptive target (gaps=infeasible)",
    )
    for ax, label in zip(axes, ["Power (kW)", "Battery energy (kWh)", "Unserved power (kW)"]):
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "dispatch.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Islanded microgrid research workbench")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "demo"]:
        p = sub.add_parser(name)
        p.add_argument("--output", default=f"outputs/{name}")
        p.add_argument("--csv", help="Hourly UTC timestamp,pv_kw,wind_kw,load_kw CSV")
        p.add_argument("--hours", type=int, default=2160)
        p.add_argument("--seed", type=int, default=42)
        if name == "demo":
            p.add_argument("--config", default=None)
            p.add_argument("--dispatch-hours", type=int, default=72)
            p.add_argument("--trees", type=int, default=60)
            p.add_argument("--scenarios", type=int, default=100)
            p.add_argument("--alpha", type=float, default=0.05)
    p = sub.add_parser("train-agent")
    p.add_argument("--data", required=True, help="Prepared hourly Parquet")
    p.add_argument("--output", default="outputs/agents/vanilla")
    p.add_argument(
        "--variant", choices=["vanilla", "fixed", "adaptive", "constrained"], default="vanilla"
    )
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reserves", help="NPY reserves for the selected training interval only")
    p = sub.add_parser("evaluate-agent")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", default="outputs/agents/evaluation")
    p.add_argument(
        "--variant", choices=["vanilla", "fixed", "adaptive", "constrained"], default="vanilla"
    )
    p.add_argument("--reserve-json", help="reserves.json from matching demo run")
    p.add_argument("--hours", type=int, default=72)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.csv, args.hours, args.seed)
    elif args.command == "demo":
        demo(args)
    elif args.command == "train-agent":
        from .agents import train, training_inputs

        frame = pd.read_parquet(args.data)
        audit(frame)
        training = frame.iloc[splits(len(frame))["train"]].reset_index(drop=True)
        training, reserve = training_inputs(training, args.variant, args.seed)
        if args.reserves:
            reserve = np.load(args.reserves)
        train(training, args.output, args.variant, args.steps, args.seed, reserve)

    else:
        from .agents import evaluate

        frame = pd.read_parquet(args.data)
        audit(frame)
        selected = frame.iloc[splits(len(frame))["test"]].iloc[: args.hours].reset_index(drop=True)
        reserve = None
        if args.reserve_json:
            rows = json.loads(Path(args.reserve_json).read_text())
            if len(rows) < len(selected) or any(
                pd.Timestamp(r["timestamp"]) != ts for r, ts in zip(rows, selected.timestamp)
            ):
                raise ValueError("Reserve timestamps do not match evaluation data")
            reserve = np.array(
                [
                    np.nan if r["end_of_step_infeasible"][0] else r["end_of_step_targets_kwh"][0]
                    for r in rows[: len(selected)]
                ]
            )
        print(
            json.dumps(evaluate(args.model, selected, args.output, args.variant, reserve), indent=2)
        )


if __name__ == "__main__":
    main()
