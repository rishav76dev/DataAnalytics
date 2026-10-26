# Recorded results

See [MPC_RESEARCH.md](MPC_RESEARCH.md) for the frozen 2025 degradation-aware MPC study. It does **not** establish robust matched-reliability ageing superiority; full hourly traces are committed. The measurements below belong to the earlier synthetic engineering demonstration and have not been replaced with newer results.

Recorded on 2026-10-24 from executed cloud runs. These results use synthetic data and describe engineering validation, not real-world research performance.

## Run configuration

- Synthetic dataset: 2,160 hourly samples, seed 42.
- Chronological split: 60% training / 15% validation / 10% calibration / 15% test.
- Evaluated episode: first 72 hours of the test partition.
- Battery: 100 kWh nominal, 50 kW, 10–90% SOC, 50% initial SOC, fixed SOH 1.0, 95% charge/discharge efficiency.
- Forecast horizon: 24 hours; nine LightGBM quantile models; 100 scenarios per origin; risk alpha 0.05.
- Total demand in the evaluated episode: 865.381 kWh.

## Verification

| Check | Executed outcome |
|---|---|
| Automated suite with RL installed | 18 passed |
| Ruff lint and formatting | Passed |
| Seven-controller synthetic demo | Completed |
| Four SAC variants | Each trained 1,500 steps, saved and evaluated |
| Adaptive individual training/evaluation CLI | Completed |
| 24-hour benchmark sweep entry point | Completed |

Gymnasium issued advisory warnings for unbounded observation-space declarations and missing registered render specification. These were warnings, not failed tests.

## Dispatch benchmark

| Controller | EENS (kWh) | Energy LPSP (%) | Curtailed (kWh) | DC throughput (kWh) |
|---|---:|---:|---:|---:|
| rule_based | 97.342 | 11.25 | 218.678 | 498.924 |
| deterministic_mpc | 97.587 | 11.28 | 218.949 | 498.409 |
| fixed_reserve_mpc | 114.477 | 13.23 | 237.664 | 462.851 |
| adaptive_reserve_mpc | 119.120 | 13.77 | 242.102 | 453.747 |
| independent_reserve_mpc | 113.823 | 13.15 | 236.939 | 464.228 |
| scenario_mpc | 97.342 | 11.25 | 228.993 | 489.125 |
| perfect_foresight_non_deployable | 97.342 | 11.25 | 279.626 | 441.024 |

Perfect foresight uses future actual values and is non-deployable. Scenario MPC has a common first action but optimistic scenario-specific later recourse. Different curtailment and throughput at equal EENS can reflect dispatch timing and remaining terminal energy; equal EENS does not imply equal trajectories.

The adaptive-reserve MPC did not improve EENS over fixed reserve or the rule-based baseline in this run. Its lower throughput cannot establish an ageing benefit because reliability also worsened. No controller met the provisional 1% energy-LPSP target.

## Forecast validation

| Metric | Measured value |
|---|---:|
| MAE | 2.313 kW |
| RMSE | 4.224 kW |
| Mean pinball loss | 0.740 kW |
| Nominal 80% interval coverage | 64.32% |
| Mean interval width | 4.882 kW |

Coverage is pooled across sources and forecast horizons. The intervals undercovered; this run does not justify calibrated probabilistic reliability claims.

## Reserve feasibility and uncertainty

- The empirical 95th-percentile initial-energy reserve was infeasible at **63 of 72** dispatch origins.
- Mean scenario infeasibility across origins: **47.33%**.
- Infeasible scenarios remain in the reserve distribution. Null serialized targets preserve failure status; operational soft fallback targets do not make them feasible.
- Adaptive-minus-fixed mean hourly unserved-power difference: **0.0645 kW**.
- Paired 95% bootstrap interval: **[-0.1429, 0.2581] kW**, using 24-hour blocks and 1,000 replicates.

The interval spans zero, so this short run supplies no clear evidence of an adaptive advantage. The point estimate is worse for adaptive MPC.

## SAC integration runs

| Variant | Training steps | Seed | Held-out EENS (kWh) | Energy LPSP (%) |
|---|---:|---:|---:|---:|
| vanilla | 1500 | 42 | 101.862 | 11.77 |
| fixed | 1500 | 42 | 97.828 | 11.30 |
| adaptive | 1500 | 42 | 97.828 | 11.30 |
| constrained | 1500 | 42 | 97.828 | 11.30 |

These are short integration runs, not converged or tuned policies. Equal outcomes in three variants do not establish equivalence of the algorithms. One seed does not establish stability or constrained-policy compliance.

## Reproduce and inspect

Follow [SETUP.md](SETUP.md) and [EVALUATION.md](EVALUATION.md). Exact input hashes recorded for the default demo:

```text
data_sha256  b3740b371b49009af9fc236a66b0dec8655889db77a6e80ac982439473d92314
model_sha256 cb40bbee56a8c63651ff14ef7ff6143e91deb9915d01a9f871ba419833e12d03
```

The full manifest includes the battery, evaluation boundaries, seed, scenario count and risk. Runtime measurements vary with hardware and model serialization may vary across platforms. Generated JSON/CSV/Parquet/PNG outputs and checkpoints remain under ignored `outputs/`; a fresh clone must regenerate them. This document retains the measured summary in version control.

## Remaining evidence

No public-data experiment, verified LFP ageing parameterization, matched-reliability degradation study, complete RL ablation matrix or conference-ready empirical result has been completed. The current ageing model is explicitly illustrative. See [docs/phase_status.md](docs/phase_status.md) for implementation and research completion separately.
