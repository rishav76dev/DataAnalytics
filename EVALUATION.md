# Evaluation guide

The later constructed Indian reference study evaluates 3,947 test hours and three SAC seeds per variant. See [reproduction steps](docs/india_research_execution.md) and [executed results](INDIA_RESULTS.md). The synthetic results and commands below remain available as the original engineering baseline. The expanded suite now has **30 tests**; the original suite described below had 18.

Use the same audited data, battery configuration, initial SOC, chronology and physical action projection when comparing controllers. The default workflow is a synthetic engineering check; an empirical study additionally requires the controls in [docs/research_protocol.md](docs/research_protocol.md).

## Automated checks

From the repository root:

```bash
uv run --frozen --extra dev --extra rl pytest -q
uv run --frozen --extra dev --extra rl ruff check src tests scripts
```

The original installed suite contained 18 tests; later India, published-ageing, capacity-feedback and greedy-oracle checks expand it to 30. Tests cover battery energy conservation and constraints, Gymnasium behavior, data continuity, forecast causality, synchronized residual blocks, MPC dispatch, infeasibility-aware reserve quantiles, reserve recursion versus an independent LP oracle, rainflow counting, bootstrap behavior and SAC checkpoint reload. The SAC test skips when optional RL dependencies are absent; distinguish that outcome from a full pass.

## End-to-end controller benchmark

```bash
uv run microgrid demo --seed 42 --hours 2160 --dispatch-hours 72 \
  --scenarios 100 --alpha 0.05 --output outputs/demo
```

The chronological split is 60% training, 15% validation, 10% calibration and 15% test. Only the first 72 test hours are dispatched by default. Forecast fitting uses training only; forecast metrics use validation; residual blocks use completed calibration origins. At dispatch, current PV/wind/load are measured and shared across scenarios. Future measurements appear only in the labelled perfect-foresight diagnostic.

Compared controllers:

1. Rule-based surplus charging and deficit discharging.
2. Deterministic central-forecast MPC.
3. Fixed-reserve MPC.
4. Joint-scenario adaptive-reserve MPC.
5. Independent-residual adaptive-reserve MPC ablation.
6. Shared-first-stage, two-stage scenario MPC.
7. Complete-episode perfect foresight, explicitly non-deployable.

Scenario MPC allows scenario-specific recourse after the first action. It is an optimistic two-stage benchmark, not a full multistage scenario tree.

## Metrics and artifacts

| Artifact | What to inspect |
|---|---|
| `data/audit.json` | Provenance, units, continuity and split boundaries |
| `frozen_manifest.json` | Data/model hashes, battery settings and evaluation boundaries recorded before metrics |
| `forecast_validation.json` | MAE, RMSE, pinball loss, nominal 80% interval coverage and width |
| `reserves.json` | Reserve targets, infeasible probability mass, power/capacity failure counts and aligned end-of-step targets |
| `metrics.json`, `comparison.csv` | EENS, energy LPSP, curtailment, throughput and runtime |
| `trace_*.parquet` | Hourly SOC, energy, powers, curtailment and unserved load |
| `paired_bootstrap.json` | Adaptive-minus-fixed mean hourly unserved-power difference and paired interval |
| `shortage_rule_based.json` | Separate reduced-generation stress diagnostic |
| `dispatch.png` | Generation, load, energy targets and unserved-load trajectories |

EENS is the sum of unserved power × timestep, in kWh. Energy LPSP is EENS divided by demand energy; it is not the fraction of hours with an outage. Loss-of-load hours are reported separately. Positive bootstrap differences mean adaptive MPC served less demand than fixed-reserve MPC. For the default hourly data, multiply the bootstrap mean/interval by evaluated hours to express total EENS differences.

The bootstrap uses 24-hour circular paired blocks and 1,000 replicates. A three-day evaluation supplies limited statistical information. Runtime contains controller-only measurements and a separately labelled shared forecast/scenario/reserve preparation time; shared preprocessing is conservative for central-forecast controllers. Do not present controller-only latency as end-to-end cost.

## SAC training and evaluation

Run all four integration checks:

```bash
uv run --extra rl python scripts/rl_smoke.py --steps 1500 --seed 42
```

Or train and evaluate one variant:

```bash
uv run --extra rl microgrid train-agent \
  --data outputs/demo/data/hourly.parquet --variant adaptive \
  --steps 10000 --seed 42 --output outputs/agents/adaptive

uv run --extra rl microgrid evaluate-agent \
  --data outputs/demo/data/hourly.parquet --model outputs/agents/adaptive \
  --variant adaptive --reserve-json outputs/demo/reserves.json \
  --hours 72 --output outputs/agents/adaptive_evaluation
```

Variants are `vanilla`, `fixed`, `adaptive`, and `constrained`. Outer training is subdivided into forecasting fit, residual calibration and a common policy-training suffix so adaptive reserves remain causal. Evaluation reserves must match evaluation timestamps. The supplied smoke-run script assumes the default battery configuration and matching default demo artifacts.

Use validation for training-length selection and hyperparameters. Repeat training across seeds and report variability before claiming stability. The constrained agent's dual update is a learned penalty, not a hard reliability guarantee. Physical battery limits remain hard.

## Capacity, risk and seed sweeps

```bash
uv run python scripts/run_benchmarks.py \
  --capacities 100 200 --alphas 0.05 0.1 --seeds 42 43 44 --hours 72
```

This runner generates synthetic optimization benchmarks for each combination; it does not train multiple RL seeds. Outputs are stored under `outputs/sweep/`. Repeated test inspection must not be used to tune a supposedly untouched final test.

## Interpreting research readiness

Report infeasible outcomes without conditioning them away. Null reserve targets mean the requested empirical risk threshold cannot be satisfied by the sampled scenarios. Check feasibility against perfect foresight before assigning deficits to learning quality. Compare degradation at matched reliability, because serving less load can also reduce battery throughput. Current ageing losses are illustrative and unvalidated; rainflow correctness does not validate LFP lifetime parameters.

A completed study still needs licensed real data, verified LFP parameters, forecaster calibration, converged multi-seed RL, complete ablations and a frozen empirical final test. Recorded engineering outcomes are in [RESULTS.md](RESULTS.md).
