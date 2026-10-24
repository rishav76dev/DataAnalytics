# Adaptive Energy-Reserve Microgrid Workbench

A runnable Python research implementation for islanded **PV + wind + LFP-style battery + demand** dispatch. Implements physical simulation, conventional optimization, quantile forecasting, joint empirical scenarios, a validated reserve recursion, and SAC training/evaluation entry points.

**Research status:** synthetic engineering demonstration. This repository does not yet establish a novel method, calibrated LFP lifetime, or a publishable advantage. The demo deliberately reports infeasibility and negative results. No public dataset was specified in the supplied plan; no dataset licence or provenance has been invented.

## Guides

- [Setup and installation](SETUP.md)
- [Evaluation and reproduction](EVALUATION.md)
- [Recorded results and limitations](RESULTS.md)

## Run

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required. From this checkout:

```bash
uv sync --frozen --extra dev --extra rl
uv run --frozen --extra dev --extra rl pytest -q
uv run --frozen --extra dev --extra rl microgrid demo --output outputs/demo
```

The default demo generates 2,160 synthetic hours, trains nine LightGBM quantile models, calibrates joint residual trajectories, and evaluates 72 held-out dispatch hours with 100 scenarios per hourly origin. It requires no credentials, external services, GPU, or data download. CPU-only PyTorch is selected explicitly. For a smaller non-RL installation omit `--extra rl`.

Outputs in `outputs/demo/`:

- `data/hourly.parquet` and `data/audit.json`: hourly data and chronological split/provenance report.
- `forecaster.pkl`, `calibration_residuals.npy`, `forecast_validation.json`: model and uncertainty diagnostics. Load pickle/checkpoint files only from trusted sources.
- `frozen_manifest.json`: data/model hashes, configuration and evaluation boundary written before controller evaluation.
- `reserves.json`: initial reserve, infeasible fraction, power/energy infeasibility, and time-aligned end-of-step targets.
- `metrics.json`, `comparison.csv`, `trace_*.parquet`: reliability, curtailment, throughput, illustrative ageing, runtime and full dispatch traces.
- `paired_bootstrap.json`, `shortage_rule_based.json`, `dispatch.png`: paired block intervals, shortage diagnostic and visualization.

## Historical data

```bash
uv run microgrid prepare --csv /path/hourly.csv --output outputs/historical
uv run microgrid demo --csv /path/hourly.csv --output outputs/historical_demo
```

CSV requires `timestamp,pv_kw,wind_kw,load_kw`. Timestamps must be strictly increasing, continuous hourly UTC timestamps; powers must be finite, nonnegative **kW**, not irradiance or energy. Missing hours are rejected instead of silently imputed. Archive original provenance, licence and unit definitions before research use. `data.py` also exposes documented simplified weather-to-power conversions and training-only capacity fitting. These are explicit library utilities; the CSV command does not silently rescale measured power.

Splits are chronological **60% training / 15% validation / 10% calibration / 15% final test**. Features use only measurements before forecast origin. Fit targets end before training boundary. Completed residual blocks end before evaluation starts. Current bus powers are assumed observed at the start of dispatch; each scenario's first stage uses that same observation. Future actual values appear only in the explicitly non-deployable perfect-foresight reference.

## SAC

```bash
# Run four short integration training/evaluation jobs, seed 42.
uv run --extra rl python scripts/rl_smoke.py --steps 1500

# Longer individual training; choose steps on validation, not final-test outcomes.
uv run --extra rl microgrid train-agent \
  --data outputs/demo/data/hourly.parquet --variant adaptive \
  --steps 10000 --output outputs/agents/adaptive
uv run --extra rl microgrid evaluate-agent \
  --data outputs/demo/data/hourly.parquet --model outputs/agents/adaptive \
  --variant adaptive --reserve-json outputs/demo/reserves.json
```

Variants: `vanilla`, `fixed`, `adaptive`, `constrained`. All use the same physically projected action and policy-training chronology. To avoid training a reserve predictor on future policy-training observations, the outer training split is further divided into 60% forecasting fit, 20% residual calibration and 20% policy training. The same final suffix is used for every variant. Adaptive training reserves are automatically computed from the nested historical model, not copied from final-test forecasts. User-provided `--reserves` is an advanced override whose causality is the caller's responsibility.

Constrained SAC uses a projected nonnegative dual multiplier updated after each episode. A learned penalty does not guarantee the reliability constraint. Saved policies, monitor CSVs and separate held-out evaluation metrics are local ignored outputs. The constrained wrapper also saves its final multiplier. More seeds, convergence analysis and model selection remain required.

## Benchmark sweep

```bash
uv run python scripts/run_benchmarks.py \
  --capacities 100 200 --alphas 0.05 0.1 --seeds 42 43 44
uv run --extra dev ruff check src tests scripts
```

This is an explicit synthetic engineering sweep, not a preregistered research result. Do not tune on the reported final test. See [research protocol](docs/research_protocol.md), [architecture](ARCHITECTURE.md), and [phase status](docs/phase_status.md) for mathematical assumptions and remaining work. `configs/system.yaml` can be supplied to `demo --config`; `configs/experiments.yaml` is the proposed study specification, not an automatically executed full benchmark matrix. `configs/battery.yaml` records illustrative sensitivity assumptions; ageing uses the matching `AgeingParameters` defaults unless a caller passes explicit parameters.

## Tests

Tests check randomized AC/DC energy conservation, SOC/SOH/power bounds, Gymnasium semantics, hourly data auditing, forecast causality, block dependence, LP first-stage dispatch, reserve quantiles including infeasible probability mass, reserve recursion against an independent LP oracle, known rainflow cycles, paired bootstrap behavior, and SAC checkpoint reload when RL dependencies are installed.
