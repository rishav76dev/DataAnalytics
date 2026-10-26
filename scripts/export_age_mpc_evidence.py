"""Audit and publish frozen computational evidence; never select test parameters."""

import hashlib
import importlib.metadata
import json
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from microgrid.age_mpc import wear_coefficients
from microgrid.physics import BatteryConfig

ROOT = Path("outputs/age_mpc_v2")
DEST = Path("docs/evidence/age_mpc_v2")
LABELS = {
    "rule_based": "Greedy reliability reference",
    "fixed_age": "Tuned fixed-reserve ageing MPC",
    "adaptive_age": "Joint adaptive-reserve ageing MPC",
    "adaptive_independent": "Independent adaptive reserve (primary comparator)",
    "scenario_envelope": "Scenario MPC + matched envelope",
    "fixed_persistence": "Tuned fixed reserve + weekly forecast",
    "adaptive_no_calendar": "Adaptive, calendar cost removed",
    "adaptive_raw": "Adaptive, uncorrected envelope",
    "deterministic_age": "Point-forecast ageing MPC",
    "persistence_age": "Weekly-forecast ageing MPC",
    "scenario": "Scenario ageing MPC",
}


def load(path):
    return json.loads(Path(path).read_text())


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_trace(t, m, coupled=False):
    power = t.battery_kw.to_numpy()
    capacity = m["capacity_kwh"] * t.physical_soh.to_numpy()
    previous = np.r_[m["initial_energy_kwh"], t.energy_kwh.to_numpy()[:-1]]
    expected = previous - np.where(power >= 0, power / 0.95, power * 0.95)
    expected -= t.capacity_fade_energy_loss_kwh.to_numpy()
    balance = t.pv_kw + t.wind_kw + power + t.unserved_kw - t.load_kw - t.curtailment_kw
    errors = {
        "max_dc_ledger_error_kwh": float(np.max(np.abs(expected - t.energy_kwh))),
        "max_ac_balance_error_kw": float(np.max(np.abs(balance))),
        "max_soc_energy_error_kwh": float(np.max(np.abs(t.soc * capacity - t.energy_kwh))),
        "operational_eens_error_kwh": abs(
            float(t.loc[~t.normalization].unserved_kw.sum()) - m["eens_kwh"]
        ),
        "damage_replay_error": abs(
            float(t.damage_25c_24h.sum()) - m["ageing"]["25c_24h"]["total_loss_fraction"]
        ),
    }
    assert max(errors.values()) < 1e-7, errors
    assert ((t.soc >= 0.1 - 1e-9) & (t.soc <= 0.9 + 1e-9)).all()
    assert np.abs(power).max() <= 50 + 1e-7
    assert (t.unserved_kw >= -1e-9).all() and (t.curtailment_kw >= -1e-9).all()
    assert (t.loc[t.normalization].battery_kw <= 1e-9).all()
    assert t.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    assert len(t) == 4200 and int(t.normalization.sum()) == 24
    if coupled:
        assert m["terminal_full_current_capacity"]
    else:
        assert m["terminal_full"]
    return errors


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    frozen = load(ROOT / "frozen_protocol.json")
    for path, expected in frozen["code_sha256"].items():
        assert sha(path) == expected, path
    assert sha("configs/age_mpc/protocol.yaml") == frozen["protocol_sha256"]
    assert sha("data/india/age_mpc_v2/hourly.parquet") == frozen["data_sha256"]
    for name, expected in frozen["model_hashes"].items():
        assert sha(ROOT / name) == expected, name
    receipts = load(ROOT / "freeze_receipts.json")
    assert sha(ROOT / "frozen_protocol.json") == receipts["frozen_protocol_sha256"]
    assert sha(ROOT / "frozen_source.zip") == receipts["frozen_source_sha256"]
    metrics = load(ROOT / "test_metrics.json")
    matched = load(ROOT / "matched_reliability.json")
    checks, traces, rows, sensitivity = {}, {}, [], []
    coefficient = wear_coefficients(BatteryConfig(capacity_kwh=400, power_kw=50))
    proxy = []
    for name in LABELS:
        m = metrics[name]
        trace = ROOT / "test" / f"trace_{m['candidate']['name']}.parquet"
        t = pd.read_parquet(trace)
        traces[name] = t
        checks[name] = verify_trace(t, m)
        shutil.copy2(trace, DEST / f"trace_{name}.parquet")
        age = m["ageing"]["25c_24h"]
        eligible = name == "rule_based" or (
            matched["comparisons"][name]["bootstrap_noninferiority"]
            and matched["comparisons"][name]["endpoint_match"]
        )
        rows.append(
            {
                "controller": name,
                "label": LABELS[name],
                "operational_eens_kwh": m["eens_kwh"],
                "operational_energy_lpsp_percent": 100 * m["lpsp_energy"],
                "normalization_eens_kwh": m["normalization_eens_kwh"],
                "all_hours_eens_kwh": float(t.unserved_kw.sum()),
                "all_hours_energy_lpsp_percent": 100 * float(t.unserved_kw.sum() / t.load_kw.sum()),
                "model_capacity_loss_percent": 100 * age["total_loss_fraction"],
                "calendar_loss_percent": 100 * age["calendar_loss_fraction"],
                "cycling_loss_percent": 100 * age["cycling_loss_fraction"],
                "rainflow_efc": age["rainflow_efc"],
                "operational_dc_throughput_kwh": m["throughput_kwh"],
                "all_hours_dc_throughput_kwh": float(t.throughput_kwh.sum()),
                "mean_soc": float(t.soc.mean()),
                "surplus_mean_soc": m["surplus_mean_soc"],
                "reserve_infeasible_fraction": m["infeasible_fraction"],
                "controller_p95_ms": m["controller_p95_ms"],
                "amortized_pipeline_seconds": m["amortized_end_to_end_seconds"],
                "matched_to_greedy": eligible,
                "out_of_fit_dod_windows": age["out_of_fit_dod_windows"],
                "ageing_windows": age["window_count"],
            }
        )
        # Post-test diagnostic only: a linear surrogate is not the offline model.
        proxy.append(
            [
                name,
                float(
                    coefficient["calendar_per_kwh_hour"] * t.energy_kwh.sum()
                    + coefficient["cyclic_per_dc_kwh"] * t.throughput_kwh.sum()
                ),
                age["total_loss_fraction"],
                eligible,
            ]
        )
        for key, loss in m["ageing"].items():
            sensitivity.append(
                {
                    "controller": name,
                    "setting": key,
                    "model_capacity_loss_percent": 100 * loss["total_loss_fraction"],
                    "out_of_fit_dod_windows": loss["out_of_fit_dod_windows"],
                    "windows": loss["window_count"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(DEST / "controller_results.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(DEST / "ageing_sensitivity.csv", index=False)
    pd.DataFrame(
        proxy, columns=["controller", "linear_wear_proxy", "offline_model_loss", "qualified"]
    ).to_csv(DEST / "surrogate_diagnostic.csv", index=False)
    correlations = {}
    for label, group in [("all", proxy), ("reliability_qualified", [r for r in proxy if r[3]])]:
        correlations[label] = {
            "n": len(group),
            "spearman": float(spearmanr([r[1] for r in group], [r[2] for r in group]).statistic),
        }
    correlations["scope"] = (
        "Post-test ranking diagnostic on this case; no coefficient fit, causal claim or empirical degradation validation"
    )
    dump(DEST / "surrogate_diagnostic.json", correlations)

    coupled = load(ROOT / "coupled_metrics.json")
    for name, m in coupled.items():
        source = (
            ROOT
            / f"coupled_{int(m['physical_temperature_c'])}c"
            / f"trace_{m['candidate']['name']}.parquet"
        )
        checks[f"coupled_{name}"] = verify_trace(pd.read_parquet(source), m, coupled=True)
        shutil.copy2(source, DEST / f"trace_coupled_{name}.parquet")
    dump(DEST / "trace_verification.json", checks)
    summary = matched["proposed_vs_strong_baselines"]
    gates = {}
    for name, comparison in summary.items():
        direct_match = all(
            v["EENS"]["ci95"][1]
            <= matched["margin_kwh"] / metrics["adaptive_age"]["operational_hours"]
            for v in comparison["paired_intervals"].values()
        )
        age_confident = all(
            v["damage"]["ci95"][1] < 0 for v in comparison["paired_intervals"].values()
        )
        passed = (
            comparison["both_match_reference_reliability_and_inventory"]
            and direct_match
            and age_confident
            and comparison["practical_reduction_at_least_one_percent"]
            and comparison["stable_damage_sign"]
        )
        gates[name] = {
            "direct_paired_reliability_noninferiority": direct_match,
            "both_match_greedy": comparison["both_match_reference_reliability_and_inventory"],
            "damage_ci_below_zero_both_blocks": age_confident,
            "practical_reduction_at_least_one_percent": comparison[
                "practical_reduction_at_least_one_percent"
            ],
            "sensitivity_sign_stable": comparison["stable_damage_sign"],
            "computational_superiority_gate": passed,
        }
    dump(
        DEST / "claim_gates.json",
        {
            "primary_comparator": frozen["primary_comparator"],
            "comparisons": gates,
            "pack_applicability_validated": False,
            "ieee_novelty_established": False,
            "conclusion": "No robust matched-reliability ageing superiority established",
        },
    )

    for name in [
        "frozen_protocol.json",
        "freeze_receipts.json",
        "development_data_manifest.json",
        "validation_calibration_summary.json",
        "test_calibration_summary.json",
        "development_forecast_report.json",
        "forecast_future_diagnostics.json",
        "test_metrics.json",
        "matched_reliability.json",
        "coupled_metrics.json",
        "coupled_serialization_receipt.json",
        "coupled_runner_at_execution.py",
        "test_forecast_runtime.json",
        "frozen_source.zip",
    ]:
        shutil.copy2(ROOT / name, DEST / name)
    shutil.copy2("data/india/age_mpc_v2/audit.json", DEST / "data_audit.json")
    for name in ["datacite.json", "files_manifest.json"]:
        shutil.copy2(Path("data/india/primary") / name, DEST / name)
    # Full per-origin coverage and reserve requests are small, reviewable evidence.
    for name in ["test_calibration.json", "test_independent_calibration.json"]:
        shutil.copy2(ROOT / name, DEST / name)
    for directory in ["weather", "weather_2025"]:
        shutil.copy2(
            Path("data/india") / directory / "weather_manifest.json",
            DEST / f"{directory}_manifest.json",
        )
    versions = {
        "python": sys.version,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "scipy", "pandas", "pyarrow", "lightgbm", "rainflow", "openpyxl"]
        },
    }
    dump(DEST / "environment_versions.json", versions)
    coupled_receipt = load(DEST / "coupled_serialization_receipt.json")
    assert sha(DEST / "coupled_runner_at_execution.py") == coupled_receipt["wrapper_sha256"]

    # Scientific figures are standalone artifacts, reproducible from released traces.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    chosen = table.iloc[:6]
    labels = [
        "Greedy",
        "Fixed",
        "Joint adaptive",
        "Independent",
        "Scenario + envelope",
        "Fixed + weekly",
    ]
    axes[0].barh(
        labels,
        chosen.model_capacity_loss_percent,
        color=["#888888", "#5278a3", "#d47737", "#4e9672", "#5278a3", "#5278a3"],
    )
    axes[0].set_xlabel("Predicted capacity loss (%) · 25°C / 24h")
    axes[0].set_title("175 days, model predictions only")
    axes[1].barh(labels, chosen.operational_eens_kwh, color="#5278a3")
    axes[1].set_xlabel("Operational EENS (kWh) · 4,176 hours")
    axes[1].set_title("Independent baseline misses reliability margin")
    fig.savefig(DEST / "matched_comparison.png", dpi=180)
    plt.close(fig)
    cal = matched["calibration"]["by_month"]
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    ax.plot(
        list(cal), [v["raw_joint_path_coverage"] for v in cal.values()], "o-", label="Raw envelope"
    )
    ax.plot(
        list(cal),
        [v["calibrated_joint_path_coverage"] for v in cal.values()],
        "o-",
        label="Delayed rolling correction",
    )
    ax.axhline(0.95, ls="--", color="black", label="Nominal 95%")
    ax.axhline(0.90, ls=":", color="gray", label="Aggregate study gate 90%")
    ax.set(
        ylabel="Simultaneous deficit-path coverage",
        ylim=(0, 1.04),
        title="Temporal undercoverage remains in February",
    )
    ax.legend(fontsize=8)
    fig.savefig(DEST / "calibration.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Degradation-aware adaptive-reserve MPC: frozen 2025 results",
        "",
        "**Conclusion: no robust matched-reliability ageing superiority is established.** The computational study is complete; empirical battery applicability and IEEE novelty are not established. Further SAC training remains paused.",
        "",
        "## Test and data",
        "",
        "The fresh holdout is January 1–June 24, 2025: 4,200 hours, including a common final 24-hour recharge-only normalization day. Operational reliability uses 4,176 hours and 58,652.039 kWh demand. All controllers finish at 360 kWh; normalization causes 9.1599 kWh additional unserved demand for every controller, reported separately. Ageing includes all 4,200 hours. The normalization policy is an experimental boundary condition and contributes to the results.",
        "",
        "The constructed reference uses Jaisalmer NASA POWER weather and scaled Indian national demand. It is not a measured local microgrid. Training-only scaling gives 95.653 kW PV and 126.764 kW wind; validation selects a 400 kWh / 50 kW battery, 10–90% SOC and 95% charge/discharge efficiencies. Primary-source verification now confirms all 12,984 demand hours in the 2024–2025 workbook against the mirror, publisher SHA-256, units, attribution and CC BY 4.0. Naive timestamp interpretation, earlier mirror provenance, simplified wind conversion and site representativeness remain limitations.",
        "",
        "## Frozen comparison",
        "",
        "| Controller | Operational EENS (kWh) | Energy LPSP (%) | Predicted capacity loss (%) | Matches greedy reliability |",
        "|---|---:|---:|---:|:---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['operational_eens_kwh']:.3f} | {r['operational_energy_lpsp_percent']:.4f} | {r['model_capacity_loss_percent']:.4f} | {'Yes' if r['matched_to_greedy'] else 'No'} |"
        )
    lines += [
        "",
        "Capacity loss uses the published BLAST-Lite LFP-Gr 250Ah cell model at assumed 25°C with exact effective-age integration and 24-hour updates. These are predictions, not measured loss or lifetime extension. Reliability matching requires both daily and weekly paired block-bootstrap upper EENS bounds within 0.01 percentage points of energy LPSP, plus equal terminal inventory.",
        "",
        "Joint adaptive reserve predicts **1.78% lower loss than tuned fixed reserve** (1.2140% versus 1.2360% capacity loss), with zero operational EENS for both. The daily paired damage interval excludes zero; the weekly interval includes zero. Therefore this apparent improvement fails the predeclared confidence requirement. The scenario-plus-envelope baseline has nearly identical damage (only 0.067% relative difference), with both damage intervals including zero.",
        "",
        "The independently calibrated reserve was the best validation-qualified ageing baseline and was frozen as the **primary comparator**. On the test it predicts lower loss than the proposed method but incurs 12.308 kWh operational EENS, exceeding the 5.865 kWh matching margin. The primary comparison therefore does not establish matched reliability or an ageing benefit. The primary comparator is retained rather than replaced after inspecting the test. Low-damage point/scenario policies have much worse reliability and cannot support an ageing benefit claim.",
        "",
        "Paired intervals resample frozen hourly damage increments in 24/168-hour blocks, 1,000 replicates. They describe conditional variability within this constructed episode; they do not include cell-parameter uncertainty or represent independent site replications. See [machine-readable comparisons](docs/evidence/age_mpc_v2/matched_reliability.json) and [claim gates](docs/evidence/age_mpc_v2/claim_gates.json).",
        "",
        "![Matched comparison](docs/evidence/age_mpc_v2/matched_comparison.png)",
        "",
        "## Calibration and forecast findings",
        "",
        "The revised sampler preserves synchronized seasonal/day-phase residual trajectories, enforces PV/wind nameplate limits and physical night masking, and avoids division by collapsing forecast interval widths. A delayed rolling score corrects every contiguous DC-deficit interval across the next 23 hours. This calibrates an envelope, not the joint probabilities of the draws.",
        "",
        "Fresh-test simultaneous coverage rises from **19.34% raw to 91.26% corrected**, exceeding the predeclared aggregate 90% gate but missing the nominal 95% target. February coverage is **70.39%**, so temporal calibration remains inadequate under drift. Forecast diagnostics by variable and lead are released separately; marginal intervals are not used as controller safety guarantees. Weekly persistence remains a strong benchmark. No claim of fully calibrated scenarios or learned-forecast superiority is made.",
        "",
        "![Calibration by month](docs/evidence/age_mpc_v2/calibration.png)",
        "",
        "## Battery verification and sensitivity",
        "",
        "The implementation preserves upstream parameter/reference compatibility and adds analytic power-law integration checks, physical action projection, strict energy accounting, and source-specific temperature/rate guards. Published model parameters are unchanged. Sensitivities cover 25/35°C and 12/24/48-hour updates. Every primary adaptive evaluation window falls below the source model's 0.8–1 fitted DOD range: **100% are extrapolated**. Numerical verification is not pack validation. A different shallow-cycle-validated LFP model or independent cell/pack measurements are required before interpreting these numbers as physical ageing benefits.",
        "",
        "Frozen-policy daily capacity-fade feedback also ran at 25°C and 35°C, preserving SOC and explicitly accounting for lost stored energy. Greedy and joint adaptive retain zero operational EENS; independent reserve incurs 12.679/12.922 kWh. These are model sensitivities, not extra confirmatory tests or new parameter selection. All six runs restore the maximum inventory of their remaining capacity. A NumPy-boolean JSON serialization failure was corrected in a separate wrapper without changing frozen simulation source or numerical outputs.",
        "",
        "## Reproducibility and research decision",
        "",
        "Full primary and capacity-feedback hourly traces, calibration records, registry metadata, frozen source archive, parameter receipt, package versions and SHA-256 inventory are committed in [the evidence directory](docs/evidence/age_mpc_v2/README.md). [Protocol](docs/degradation_mpc_protocol.md), [data/model validation](docs/age_mpc_validation.md), and [reproduction commands](docs/age_mpc_reproduction.md) explain the implementation and limits. Earlier results remain historical evidence.",
        "",
        "Do not spend further SAC compute or claim IEEE-level novelty from this experiment. The next confirmatory study needs an independently sourced hourly Indian site load, measured or validated generation conversions, a shallow-cycle-applicable LFP model, drift-aware calibration developed without reusing this holdout for confirmation, and multiple untouched seasons/sites. The useful result here is a reproducible, guarded comparison with a negative/inconclusive superiority finding.",
    ]
    Path("MPC_RESEARCH.md").write_text("\n".join(lines) + "\n")
    inventory = {
        str(p.relative_to(DEST)): sha(p)
        for p in sorted(DEST.rglob("*"))
        if p.is_file() and p.name not in {"checksums.json", "review_bundle.zip"}
    }
    dump(DEST / "checksums.json", inventory)
    with zipfile.ZipFile(ROOT / "review_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(DEST.rglob("*")):
            if p.is_file():
                z.write(p, str(p.relative_to(DEST)))
    print(f"Verified {len(checks)} full traces; released {len(inventory)} evidence files")


if __name__ == "__main__":
    main()
