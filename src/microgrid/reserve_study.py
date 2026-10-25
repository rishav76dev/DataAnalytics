"""Validation-only fixed-reserve tuning and causal reserve counterfactuals."""

import json
from pathlib import Path

import numpy as np

from .controllers import deterministic_mpc, rule_based, scenario_mpc
from .data import CHANNELS, splits
from .evaluation import dispatch, paired_block_bootstrap
from .feasibility import perfect_foresight_episode
from .forecasting import QuantileForecaster, residual_blocks
from .physics import BatteryConfig
from .reserve import ReserveEstimator, end_of_step_targets
from .scenarios import sample_blocks


def validate_reserve(frame, output, hours=72, seed=42, scenarios=100):
    output = Path(output)
    c = BatteryConfig()
    parts = splits(len(frame))
    fit_end = int(0.8 * parts["train"].stop)
    model = QuantileForecaster(seed=seed, trees=30).fit(frame, fit_end, stride=24)
    # All residual outcomes precede BOTH validation tuning and evaluation.
    residual = residual_blocks(model, frame, fit_end, parts["train"].stop)
    middle = (parts["validation"].start + parts["validation"].stop) // 2
    if hours < 24 or hours > min(
        middle - parts["validation"].start, parts["validation"].stop - middle
    ):
        raise ValueError("Choose >=24 hours within each validation half")

    def inputs(start):
        forecasts, joint, targets, relaxed, diagnostics = [], [], [], [], []
        for origin in range(start, start + hours):
            q = model.predict(frame, origin)
            f = q[:, :, 1].copy()
            s = sample_blocks(q, residual, scenarios, seed + origin)
            current = frame[CHANNELS].iloc[origin].to_numpy()
            s[:, 0], f[0] = current, current
            estimator = ReserveEstimator(c)
            r = estimator.estimate(s)
            target, bad = end_of_step_targets(s, estimator)
            no_fallback = target.copy()
            no_fallback[np.array(bad)] = c.minimum
            forecasts.append(f)
            joint.append(s)
            targets.append(target)
            relaxed.append(no_fallback)
            diagnostics.append({"origin": origin, **r, "end_of_step_infeasible": bad})
        return forecasts, joint, targets, relaxed, diagnostics

    train_start = parts["validation"].start
    tuning, _, _, _, _ = inputs(train_start)
    tuning_frame = frame.iloc[train_start : train_start + hours].reset_index(drop=True)
    tuning_results = {}
    for fraction in [0.2, 0.4, 0.6, 0.8]:
        _, result = dispatch(
            tuning_frame,
            lambda t, e, x: deterministic_mpc(tuning[t], e, c, reserve_kwh=fraction * c.capacity),
            c,
        )
        tuning_results[fraction] = result
    best = min(
        tuning_results,
        key=lambda f: (tuning_results[f]["eens_kwh"], tuning_results[f]["throughput_kwh"]),
    )
    # Persist tuned choice before evaluation. No final-test outcomes inspected.
    (output / "selection.json").write_text(
        json.dumps(
            {
                "fit_stop": fit_end,
                "residual_stop": parts["train"].stop,
                "tuning_start": train_start,
                "tuning_hours": hours,
                "evaluation_start": middle,
                "evaluation_hours": hours,
                "chosen_fixed_fraction": best,
                "tuning_results": tuning_results,
                "final_test_evaluated": False,
            },
            indent=2,
        )
    )
    forecasts, joint, targets, relaxed, diagnostics = inputs(middle)
    evaluated = frame.iloc[middle : middle + hours].reset_index(drop=True)
    methods = {
        "rule_based": lambda t, e, x: rule_based(x),
        "deterministic_no_reserve": lambda t, e, x: deterministic_mpc(forecasts[t], e, c),
        "validation_tuned_fixed": lambda t, e, x: deterministic_mpc(
            forecasts[t], e, c, reserve_kwh=best * c.capacity
        ),
        "adaptive_max_fallback": lambda t, e, x: deterministic_mpc(
            forecasts[t], e, c, reserve_kwh=targets[t]
        ),
        "adaptive_no_infeasible_target": lambda t, e, x: deterministic_mpc(
            forecasts[t], e, c, reserve_kwh=relaxed[t]
        ),
        "scenario_mpc": lambda t, e, x: scenario_mpc(joint[t], e, c),
    }
    traces, results = {}, {}
    for name, controller in methods.items():
        trace, result = dispatch(evaluated, controller, c)
        result["unserved_while_end_energy_above_floor_kwh"] = float(
            trace.loc[trace.energy_kwh > c.minimum + 1e-6, "unserved_kw"].sum() * c.dt_hours
        )
        trace.to_parquet(output / f"trace_{name}.parquet", index=False)
        traces[name], results[name] = trace, result
    oracle, ref = perfect_foresight_episode(evaluated, c)
    oracle.to_parquet(output / "trace_perfect_foresight.parquet", index=False)
    results["perfect_foresight_non_deployable"] = ref
    paired = paired_block_bootstrap(
        traces["adaptive_max_fallback"].unserved_kw,
        traces["validation_tuned_fixed"].unserved_kw,
        seed=seed,
    )
    report = {
        "status": "validation_only_no_research_superiority_claim",
        "chosen_fixed_fraction": best,
        "historical_residual_blocks": len(residual),
        "results": results,
        "paired_bootstrap": paired,
        "reserve_diagnostics": diagnostics,
        "note": "Different forecaster and partition from original demo; do not compare absolute EENS across experiments. CQR marginal interval correction is not residual-scenario tail calibration.",
    }
    (output / "reserve_validation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
