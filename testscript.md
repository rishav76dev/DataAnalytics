# How to check whether the recorded results are right

Run these commands from the repository root with Python 3.12+. They check the committed frozen experiment summarized in [result.md](result.md). They do not retrain controllers or overwrite the held-out results.

## 1. Install the pinned dependencies

```bash
export UV_CACHE_DIR=/tmp/uv-cache
export UV_LINK_MODE=copy
uv sync --frozen --extra dev --extra rl
```

Installation requires network access. Once installed, the release verification below uses only committed files; ignored `data/` and `outputs/` directories are not needed. Use ordinary Python execution, not `python -O`, because the verification uses assertions.

## 2. Run the release and software checks

```bash
.venv/bin/python scripts/verify_age_mpc_release.py
MPLCONFIGDIR=/tmp/microgrid-mpl XDG_CACHE_HOME=/tmp/microgrid-cache \
  .venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
```

Expected for this release:

```text
Verified 49 evidence files, frozen source, and 11 primary traces
43 passed
All checks passed!
```

Pytest can also report Gymnasium observation-space/render-mode warnings. They are not failures. The release verifier checks SHA-256 file integrity, matching frozen numerical source and protocol, and primary trace totals. It does not by itself recompute the ageing model or statistical comparisons; the next check does that.

## 3. Recompute physical, numerical and statistical checks

Copy this complete block into a Bash terminal at the repository root. It reads all 17 traces, checks physical accounting, recalculates ageing from SOC, recomputes selected confidence intervals, and checks calibration records. It prints results and writes no files.

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

import numpy as np
import pandas as pd

from microgrid.ageing import evaluate_lfp
from microgrid.evaluation import paired_block_bootstrap, reliability

root = Path("docs/evidence/age_mpc_v2")

def load(name):
    return json.loads((root / name).read_text())

def close(actual, expected, tolerance=1e-8):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)

def check_trace(trace, metric, coupled=False):
    assert len(trace) == 4200
    assert trace.normalization.tolist() == [False] * 4176 + [True] * 24
    assert trace.timestamp.is_monotonic_increasing
    assert trace.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    numbers = trace.select_dtypes(include=["number"]).to_numpy()
    assert np.isfinite(numbers).all()
    assert trace.soc.between(0.1 - 1e-9, 0.9 + 1e-9).all()
    assert trace.physical_soh.between(0.5, 1).all()
    assert trace.battery_kw.abs().max() <= 50 + 1e-7
    assert (trace[["pv_kw", "wind_kw", "load_kw", "unserved_kw",
                   "curtailment_kw", "throughput_kwh"]] >= -1e-9).all().all()
    assert (trace.unserved_kw <= trace.load_kw + 1e-9).all()

    # Positive battery_kw is AC discharge; energy_kwh is post-step DC energy.
    power = trace.battery_kw.to_numpy()
    ac_error = (trace.pv_kw + trace.wind_kw + power + trace.unserved_kw
                - trace.load_kw - trace.curtailment_kw)
    close(ac_error, 0, 1e-7)
    previous = np.r_[metric["initial_energy_kwh"], trace.energy_kwh.to_numpy()[:-1]]
    expected = previous - np.where(power >= 0, power / 0.95, power * 0.95)
    expected -= trace.capacity_fade_energy_loss_kwh.to_numpy()
    close(trace.energy_kwh, expected, 1e-7)
    capacity = metric["capacity_kwh"] * trace.physical_soh
    close(trace.energy_kwh, trace.soc * capacity, 1e-7)
    close(trace.throughput_kwh, np.where(power >= 0, power / 0.95, -power * 0.95))

    operational = trace.loc[~trace.normalization]
    for key, value in reliability(operational).items():
        close(value, metric[key])
    close(trace.loc[trace.normalization].unserved_kw.sum(),
          metric["normalization_eens_kwh"])
    assert (trace.loc[trace.normalization].battery_kw <= 1e-9).all()
    close(trace.energy_kwh.iloc[-1], 0.9 * capacity.iloc[-1], 1e-5)
    assert metric["terminal_full_current_capacity"] if coupled else metric["terminal_full"]

    # Re-evaluate published-model predictions from SOC, not stored damage sums.
    for temperature in (25, 35):
        key = f"{temperature}c_24h"
        age = evaluate_lfp(
            np.r_[0.5, trace.soc.to_numpy()], temperature_c=temperature,
            chunk_hours=24, integration="effective_age_exact",
        )
        for field in ("calendar_loss_fraction", "cycling_loss_fraction",
                      "total_loss_fraction"):
            close(age[field], metric["ageing"][key][field], 1e-10)
        close(trace[f"damage_{key}"].sum(), age["total_loss_fraction"], 1e-10)
    if coupled:
        physical_key = f"{int(metric['physical_temperature_c'])}c_24h"
        close(1 - metric["live_final_soh"],
              metric["ageing"][physical_key]["total_loss_fraction"], 1e-8)

metrics = load("test_metrics.json")
traces = {}
for name, metric in metrics.items():
    trace = pd.read_parquet(root / f"trace_{name}.parquet")
    check_trace(trace, metric)
    traces[name] = trace
    print(f"PASS {name}: operational EENS={metric['eens_kwh']:.6f} kWh")
for name, metric in load("coupled_metrics.json").items():
    check_trace(pd.read_parquet(root / f"trace_coupled_{name}.parquet"), metric, True)
print("PASS physical ledgers and 24-hour ageing replay for all 17 traces")

matched = load("matched_reliability.json")
margin = 0.0001 * metrics["rule_based"]["demand_kwh"]
close(margin, matched["margin_kwh"])
assert matched["primary_comparator"] == "adaptive_independent"
proposed = traces["adaptive_age"]
for other in ("fixed_age", "adaptive_independent", "scenario_envelope"):
    reference = traces[other]
    saved = matched["proposed_vs_strong_baselines"][other]
    for block in (24, 168):
        pairs = {
            "EENS": (proposed.loc[~proposed.normalization, "unserved_kw"],
                     reference.loc[~reference.normalization, "unserved_kw"]),
            "damage": (proposed.damage_25c_24h, reference.damage_25c_24h),
        }
        for kind, (ours, baseline) in pairs.items():
            interval = paired_block_bootstrap(ours, baseline, block=block,
                                             repeats=1000, seed=42)
            recorded = saved["paired_intervals"][str(block)][kind]
            close(interval["mean_difference"], recorded["mean_difference"], 1e-12)
            close(interval["ci95"], recorded["ci95"], 1e-12)
    print(f"PASS paired daily/weekly intervals versus {other}")

records = pd.DataFrame(load("test_calibration.json"))
records = records.loc[records.ready]
summary = load("test_calibration_summary.json")
assert len(records) == summary["overall"]["origins"]
close(records.raw_joint_path_covered.mean(),
      summary["overall"]["raw_joint_path_coverage"])
close(records.corrected_joint_path_covered.mean(),
      summary["overall"]["calibrated_joint_path_coverage"])
months = pd.to_datetime(records.timestamp, utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m")
for month, group in records.groupby(months):
    close(group.corrected_joint_path_covered.mean(),
          summary["by_month"][month]["calibrated_joint_path_coverage"])
print("PASS aggregate and monthly calibration arithmetic")

loss = lambda name: metrics[name]["ageing"]["25c_24h"]["total_loss_fraction"]
reduction = 100 * (1 - loss("adaptive_age") / loss("fixed_age"))
weekly = matched["proposed_vs_strong_baselines"]["fixed_age"]["paired_intervals"]["168"]["damage"]["ci95"]
assert weekly[0] <= 0 <= weekly[1]
assert metrics["adaptive_independent"]["eens_kwh"] > margin
for comparison in load("claim_gates.json")["comparisons"].values():
    assert not comparison["computational_superiority_gate"]
age = metrics["adaptive_age"]["ageing"]["25c_24h"]
assert age["out_of_fit_dod_windows"] == age["window_count"] == 175
print(f"Fixed-reserve relative model-loss reduction: {reduction:.4f}%")
print(f"Reliability margin: {margin:.4f} kWh")
print("PASS recorded calculations; robust ageing superiority remains UNPROVEN")
PY
```

Expected final lines:

```text
PASS aggregate and monthly calibration arithmetic
Fixed-reserve relative model-loss reduction: 1.7825%
Reliability margin: 5.8652 kWh
PASS recorded calculations; robust ageing superiority remains UNPROVEN
```

The script deliberately expects the weekly damage interval to include zero and the primary comparator to miss the reliability margin. These are valid negative research findings, not software failures. The physical checks use the recorded hourly-mean power assumptions. The ageing replay uses the same pinned model implementation, so agreement verifies computational consistency rather than empirical battery accuracy.

## 4. Understand what each check establishes

| Check | A passing result establishes | It does not establish |
|---|---|---|
| SHA-256 inventory/source verifier | Files match the archived release and numerical source | Independent confirmation that input measurements are accurate |
| Automated tests | Tested equations, causality and controller constraints behave as specified | Correctness for every possible input |
| Full-trace energy/SOC checks | Recorded powers and battery states obey the stated physical equations | Accuracy of fixed efficiencies or weather-to-power models at a real site |
| Reliability recomputation | EENS/LPSP agree with hourly traces and explicit normalization handling | A guarantee of future reliability from one finite holdout |
| Ageing replay | Published-model outputs agree with recorded SOC and declared evaluation settings | Validated shallow-cycle pack ageing or lifetime extension |
| Paired bootstrap replay | Reported selected confidence intervals reproduce with the fixed seed | Uncertainty over battery parameters or independent site replications |
| Calibration arithmetic | Stored coverage flags aggregate to the reported percentages | A full rerun of scenario generation, or adequate nominal/conditional coverage |

For a scenario-generation and controller rerun, use [the complete reproduction guide](docs/age_mpc_reproduction.md) in a clean checkout. It requires public-source downloads and substantially more computation. Preserve the frozen evidence and compare data/source hashes before calling a new run an exact reproduction. Never tune on this holdout and present the same period as fresh confirmation.

## 5. Diagnose a failure

- **Missing package or executable:** rerun the frozen installation; run commands from the repository root.
- **Checksum/source mismatch:** inspect `git status --short` and the failing filename. Do not edit the inventory to conceal the mismatch. Intended numerical changes require a separately versioned experiment.
- **Energy or SOC assertion failure:** inspect battery sign conventions, kW versus kWh, charge/discharge efficiency and explicit capacity-fade energy loss.
- **EENS mismatch:** separate the 4,176 operational hours from the 24 normalization hours. EENS is hourly unserved power summed over one-hour steps; LPSP is EENS divided by the matching demand energy.
- **Ageing mismatch:** verify the initial SOC, temperature, update window and `effective_age_exact` integration. Do not substitute the historical illustrative model or upstream Euler evaluation.
- **Bootstrap mismatch:** retain paired timestamps, the same normalization treatment, seed 42, 1,000 replicates and 24/168-hour circular blocks.
- **All checks pass:** the released computations are internally consistent. Calibration under drift and empirical battery applicability still need further research; see [the validation card](docs/age_mpc_validation.md).
