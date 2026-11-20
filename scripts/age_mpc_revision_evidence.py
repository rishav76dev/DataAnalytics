"""Derive revision tables and a calibration plot from the immutable release.

No controller selection or dispatch is performed here. The 24-hour normalization
period is excluded from operational measures and included in the ageing replay.
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rainflow


ROOT = Path("docs/evidence/age_mpc_v2")
OUT = Path("docs/revision_evidence")
METHODS = ("rule_based", "adaptive_independent", "fixed_age", "adaptive_age", "scenario_envelope")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    controller = {r["controller"]: r for r in csv.DictReader((ROOT / "controller_results.csv").open())}
    gates = json.loads((ROOT / "matched_reliability.json").read_text())
    summary = json.loads((ROOT / "test_calibration_summary.json").read_text())
    stress = []
    comparison = []
    for name in METHODS:
        trace = pd.read_parquet(ROOT / f"trace_{name}.parquet")
        assert len(trace) == 4200 and int(trace.normalization.sum()) == 24
        op = trace.loc[~trace.normalization]
        assert len(op) == 4176
        row = controller[name]
        assert abs(op.unserved_kw.sum() - float(row["operational_eens_kwh"])) < 1e-6
        assert abs(trace.unserved_kw.sum() - float(row["all_hours_eens_kwh"])) < 1e-6
        assert abs(trace.energy_kwh.iloc[-1] - 360) < 1e-6

        # Rainflow uses SOC fraction, so depth bins are fractions of rated capacity.
        cycles = list(rainflow.extract_cycles(np.r_[0.5, trace.soc.to_numpy()]))
        count = np.array([c[2] for c in cycles], float)
        depth = np.array([c[0] for c in cycles], float)
        bin_counts = [float(count[(depth >= low) & (depth < high)].sum())
                      for low, high in ((0, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.01))]
        stress.append({
            "controller": name,
            "operational_dc_throughput_kwh": float(op.throughput_kwh.sum()),
            "all_hours_rainflow_efc": float((depth * count).sum()),
            "operational_mean_soc": float(op.soc.mean()),
            "operational_hours_soc_above_80_percent": int((op.soc > 0.8).sum()),
            "operational_fraction_soc_above_80_percent": float((op.soc > 0.8).mean()),
            "operational_mean_requested_reserve_kwh": (
                float(op.requested_reserve_kwh.mean())
                if name in {"adaptive_age", "adaptive_independent", "scenario_envelope"} else ""
            ),
            "operational_fraction_energy_at_or_above_requested_reserve": (
                float((op.energy_kwh >= op.requested_reserve_kwh - 1e-8).mean())
                if name in {"adaptive_age", "adaptive_independent", "scenario_envelope"} else ""
            ),
            "all_hours_cycle_count_dod_below_10_percent": bin_counts[0],
            "all_hours_cycle_count_dod_10_to_20_percent": bin_counts[1],
            "all_hours_cycle_count_dod_20_to_50_percent": bin_counts[2],
            "all_hours_cycle_count_dod_at_least_50_percent": bin_counts[3],
            "all_hours_cycle_count": float(count.sum()),
        })
        intervals = gates["comparisons"].get(name, {}).get("paired_intervals", {})
        comparison.append({
            "controller": name,
            "comparison_group": "primary preselected" if name == "adaptive_independent" else "reliability matched",
            "operational_eens_kwh": row["operational_eens_kwh"],
            "all_hours_eens_kwh": row["all_hours_eens_kwh"],
            "operational_lpsp_percent": row["operational_energy_lpsp_percent"],
            "operational_loss_of_load_hours": int((op.unserved_kw > 1e-9).sum()),
            "normalization_eens_kwh": row["normalization_eens_kwh"],
            "terminal_energy_kwh": float(trace.energy_kwh.iloc[-1]),
            "reserve_infeasible_fraction": row["reserve_infeasible_fraction"],
            "blast_lite_loss_percent_exploratory": row["model_capacity_loss_percent"],
            "paired_vs_greedy_eens_upper_24h_kw": intervals.get("24", {}).get("EENS_mean_hourly_difference", {}).get("ci95", [None, None])[1],
            "paired_vs_greedy_eens_upper_168h_kw": intervals.get("168", {}).get("EENS_mean_hourly_difference", {}).get("ci95", [None, None])[1],
            "paired_vs_greedy_damage_upper_24h_fraction_per_hour": intervals.get("24", {}).get("damage_mean_hourly_difference", {}).get("ci95", [None, None])[1],
            "paired_vs_greedy_damage_upper_168h_fraction_per_hour": intervals.get("168", {}).get("damage_mean_hourly_difference", {}).get("ci95", [None, None])[1],
        })
    write_csv(OUT / "battery_stress.csv", stress)
    write_csv(OUT / "matched_comparison.csv", comparison)
    direct = []
    for name in ("fixed_age", "scenario_envelope", "adaptive_independent"):
        item = gates["proposed_vs_strong_baselines"][name]
        direct.append({
            "adaptive_minus": name,
            "both_match_greedy_reliability_and_inventory": item["both_match_reference_reliability_and_inventory"],
            "relative_blast_loss_reduction": item["model_damage_reduction_fraction"],
            "paired_eens_ci95_24h_kwh_per_hour": item["paired_intervals"]["24"]["EENS"]["ci95"],
            "paired_eens_ci95_168h_kwh_per_hour": item["paired_intervals"]["168"]["EENS"]["ci95"],
            "paired_damage_ci95_24h_fraction_per_hour": item["paired_intervals"]["24"]["damage"]["ci95"],
            "paired_damage_ci95_168h_fraction_per_hour": item["paired_intervals"]["168"]["damage"]["ci95"],
        })
    write_csv(OUT / "adaptive_direct_comparisons.csv", direct)

    months = list(summary["by_month"])
    raw = [100 * summary["by_month"][m]["raw_joint_path_coverage"] for m in months]
    corrected = [100 * summary["by_month"][m]["calibrated_joint_path_coverage"] for m in months]
    x = np.arange(len(months))
    fig, ax = plt.subplots(figsize=(7.0, 3.2), layout="constrained")
    ax.bar(x - 0.18, raw, 0.36, label="Raw", color="#8fa3ad")
    ax.bar(x + 0.18, corrected, 0.36, label="Corrected", color="#175d7a")
    ax.axhline(95, color="#a44d34", linestyle="--", linewidth=1, label="Nominal 95%")
    ax.annotate("February: 70.39%", xy=(1.18, corrected[1]), xytext=(1.5, 54),
                arrowprops={"arrowstyle": "->", "color": "#a44d34"}, color="#a44d34")
    ax.set(xticks=x, xticklabels=months, ylabel="Simultaneous deficit-envelope coverage (%)", ylim=(0, 108))
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.17), frameon=False)
    fig.savefig(OUT / "monthly_coverage.pdf")
    plt.close(fig)

    print(f"Wrote revision evidence to {OUT}")


if __name__ == "__main__":
    main()
