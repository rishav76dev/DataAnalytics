"""Reporting for already frozen experiments; does not choose controller settings."""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from final_india_benchmark import dump
from scipy.stats import spearmanr

from microgrid.calibration import IntervalCalibrator, by_source_horizon
from microgrid.data import CHANNELS, splits
from microgrid.physics import BatteryConfig
from microgrid.reserve import backward_requirement
from microgrid.scenarios import sample_blocks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/india/reference/hourly.parquet")
    p.add_argument("--benchmark", default="outputs/india_study/final")
    p.add_argument("--rl", default="outputs/india_study/rl")
    args = p.parse_args()
    root = Path(args.benchmark)
    frozen = json.loads((root / "frozen_manifest.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    frame = pd.read_parquet(args.data)
    frame[["pv_kw", "wind_kw"]] *= frozen["selected_design"]["renewable_scale"]
    parts = splits(len(frame))
    c = BatteryConfig(**frozen["battery"])
    # Trusted local artifact produced in this workspace; never accept arbitrary pickle URLs.
    model = pickle.loads((root / "forecaster.pkl").read_bytes())
    residual = np.load(root / "residual_blocks.npy")
    midpoint = (parts["validation"].start + parts["validation"].stop) // 2

    def pairs(start, stop):
        origins = list(range(start, stop - 23, 24))
        truth = np.array([frame[CHANNELS].iloc[t : t + 24].to_numpy() for t in origins])
        pred = np.array([model.predict(frame, t) for t in origins])
        return origins, truth, pred

    _, truth_fit, pred_fit = pairs(parts["validation"].start, midpoint)
    calibrator = IntervalCalibrator().fit(truth_fit, pred_fit, midpoint)
    origins, truth, pred = pairs(parts["test"].start, parts["test"].stop)
    corrected = np.array([calibrator.apply(q, t) for t, q in zip(origins, pred)])
    interval_report = {
        "correction_fit_partition": "early_validation_only",
        "fit_end": midpoint,
        "correction_fit_origins": len(truth_fit),
        "test_origins": len(origins),
        "raw": by_source_horizon(truth, pred),
        "corrected": by_source_horizon(truth, corrected),
        "interpretation": "Post-run descriptive report for pre-specified correction; median/scenario dispatch unchanged; not used for test model selection",
    }
    dump(root / "forecast_test_corrected.json", interval_report)
    # Scenario diagnostics use completed observed futures, strictly for reporting.
    records = []
    for t, y, q in zip(origins, truth, pred):
        scenarios = sample_blocks(q, residual, frozen["scenario_count"], frozen["seed"] + t)
        scenarios[:, 0] = y[0]
        net = scenarios[:, :, 2] - scenarios[:, :, 0] - scenarios[:, :, 1]
        observed_net = y[:, 2] - y[:, 0] - y[:, 1]
        actual = backward_requirement(observed_net, c)
        scenarios_req = [backward_requirement(x, c).energy_kwh for x in net]
        required = np.sort(scenarios_req)
        target = required[int(np.ceil(0.95 * len(required))) - 1]
        energy95 = float(np.quantile(net.sum(axis=1), 0.95, method="higher"))
        records.append(
            {
                "origin": t,
                "actual_net_energy_kwh": float(observed_net.sum()),
                "scenario_net_energy_p95_kwh": energy95,
                "actual_net_energy_below_p95": bool(observed_net.sum() <= energy95),
                "reserve_target_finite": bool(np.isfinite(target)),
                "actual_reserve_feasible": actual.feasible,
                "actual_requirement_kwh": actual.energy_kwh if actual.feasible else None,
                "finite_target_covers_actual": bool(actual.feasible and actual.energy_kwh <= target)
                if np.isfinite(target)
                else None,
            }
        )
    finite = [r for r in records if r["reserve_target_finite"]]
    tail = {
        "origin_count": len(records),
        "historical_blocks": len(residual),
        "coverage_actual_net_energy_below_scenario_p95": float(
            np.mean([r["actual_net_energy_below_p95"] for r in records])
        ),
        "finite_target_origins": len(finite),
        "finite_target_covers_actual_fraction": float(
            np.mean([r["finite_target_covers_actual"] for r in finite])
        )
        if finite
        else None,
        "actual_energy_or_power_infeasible_origins": sum(
            not r["actual_reserve_feasible"] for r in records
        ),
        "warning": "Finite-target subset coverage is conditional, not a reliability guarantee; all infeasible cases retained in records",
        "records": records,
    }
    dump(root / "scenario_tail_diagnostics.json", tail)
    reserves = json.loads((root / "reserves.json").read_text())
    infeas = float(np.mean([r["initial_infeasible"] for r in reserves]))
    results = [
        {
            "controller": k,
            **{m: v[m] for m in ["eens_kwh", "lpsp_energy", "curtailment_kwh", "throughput_kwh"]},
            "published_loss_fraction": v["published_model_ageing"]["total_loss_fraction"],
            "terminal_energy_kwh": v["terminal_energy_kwh"],
        }
        for k, v in metrics.items()
    ]
    rl = Path(args.rl)
    if (rl / "final_metrics.json").exists():
        for name, v in json.loads((rl / "final_metrics.json").read_text()).items():
            results.append(
                {
                    "controller": "sac_" + name,
                    **{
                        m: v[m]
                        for m in ["eens_kwh", "lpsp_energy", "curtailment_kwh", "throughput_kwh"]
                    },
                    "published_loss_fraction": v["published_model_ageing"]["total_loss_fraction"],
                    "terminal_energy_kwh": v["terminal_energy_kwh"],
                }
            )
    table = pd.DataFrame(results)
    table.to_csv(root / "all_controllers.csv", index=False)
    corr = spearmanr(table.throughput_kwh, table.published_loss_fraction)
    # Post-hoc matched reliability screen, explicit tolerance; no policy selection.
    reference = metrics["rule_based"]["lpsp_energy"]
    matched = table[np.abs(table.lpsp_energy - reference) <= 0.001]
    summary = {
        "test_hours": parts["test"].stop - parts["test"].start,
        "reserve_infeasible_fraction": infeas,
        "raw_interval_coverage": interval_report["raw"]["pooled"]["coverage_80"],
        "corrected_interval_coverage": interval_report["corrected"]["pooled"]["coverage_80"],
        "scenario_net_energy_upper95_coverage": tail[
            "coverage_actual_net_energy_below_scenario_p95"
        ],
        "throughput_vs_published_loss_spearman": float(corr.statistic),
        "throughput_vs_published_loss_pvalue_descriptive_only": float(corr.pvalue),
        "matched_reliability_tolerance_absolute_lpsp": 0.001,
        "matched_to_rule_based": matched.controller.tolist(),
        "model_applicability": "Cell-level model; not calibrated to this pack. Mean25C battery assumption, daily DOD extrapolations. Offline SOH only.",
    }
    if (rl / "final_metrics.json").exists():
        rl_values = list(json.loads((rl / "final_metrics.json").read_text()).values())
        proxy_rank = spearmanr(
            [v["published_ageing_reward_proxy_total"] for v in rl_values],
            [v["published_model_ageing"]["total_loss_fraction"] for v in rl_values],
        )
        summary["rl_hourly_proxy_vs_daily_model_spearman"] = float(proxy_rank.statistic)
        summary["rl_proxy_ranking_trajectory_count"] = len(rl_values)
    dump(root / "analysis_summary.json", summary)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/microgrid-mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/microgrid-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 7))
    for r in results:
        ax.scatter(r["lpsp_energy"] * 100, r["published_loss_fraction"] * 100, s=35)
        ax.annotate(
            r["controller"],
            (r["lpsp_energy"] * 100, r["published_loss_fraction"] * 100),
            fontsize=6,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Energy LPSP (%)")
    ax.set_ylabel("Published-model capacity loss (%) — not pack-validated")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(root / "reliability_degradation.png", dpi=180)
    plt.close(fig)
    (root / "all_controllers.tex").write_text(
        table[["controller", "eens_kwh", "lpsp_energy", "published_loss_fraction"]].to_latex(
            index=False, float_format="%.6f", escape=True
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
