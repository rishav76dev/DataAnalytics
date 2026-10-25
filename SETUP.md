# Setup guide

For the completed constructed Indian reference experiment, follow [india_research_execution.md](docs/india_research_execution.md) after installing dependencies below. Its inputs, source limitations and outcomes are in [INDIA_RESULTS.md](INDIA_RESULTS.md). The default `microgrid demo` remains synthetic.

This project runs hourly islanded microgrid simulations, forecasting, reserve estimation, optimization baselines, and optional SAC training. No GPU, API keys, or background services are required for the synthetic workflow.

## Prerequisites

- Python 3.12 or later.
- [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Git and Internet access to GitHub, PyPI, and the CPU PyTorch package index during installation.

## Install

For a new machine:

```bash
git clone https://github.com/rishav76dev/DataAnalytics.git
cd DataAnalytics
uv sync --frozen --extra dev --extra rl
```

In the prepared cloud environment, use the existing checkout at `/workspace/DataAnalytics`; do not create another checkout or worktree. The lockfile fixes dependency versions. The `rl` extra installs CPU-only PyTorch and Stable-Baselines3. Omit that extra if you only need forecasts and optimization baselines.

## Verify installation

```bash
uv run --frozen --extra dev --extra rl pytest -q
uv run --frozen --extra dev --extra rl ruff check src tests scripts
```

Then execute the complete synthetic workflow:

```bash
uv run --frozen --extra dev --extra rl microgrid demo --output outputs/demo
```

The default run generates 2,160 synthetic hours, trains LightGBM quantile models, creates 100 scenarios per origin, and evaluates seven controllers over 72 held-out hours. Inspect `outputs/demo/metrics.json`, `reserves.json`, and `dispatch.png`. Successful installation alone does not validate dispatch behavior.

## Configuration

`configs/system.yaml` defines the battery and proposed scenario settings. The demo reads the `battery` section through `--config`; horizon is currently fixed at 24 hours in the forecasting implementation. Scenario count, risk and seed must be supplied using CLI flags:

```bash
uv run microgrid demo --config configs/system.yaml \
  --scenarios 100 --alpha 0.05 --seed 42 --dispatch-hours 72 \
  --output outputs/custom
```

Default battery: 100 kWh nominal capacity, 50 kW power, 10–90% SOC, 50% initial SOC, 95% charge/discharge efficiency, fixed SOH 1.0 and one-hour timesteps.

`configs/experiments.yaml` records the proposed research design; it does not automatically execute all experiments. `configs/battery.yaml` documents illustrative ageing assumptions; runtime ageing parameters are supplied through `AgeingParameters`, not automatically loaded from that YAML.

## Use historical data

Prepare a CSV with these required columns:

| Column | Meaning |
|---|---|
| `timestamp` | Continuous, increasing hourly timestamps; use explicit UTC offsets |
| `pv_kw` | Nonnegative PV generation in kW |
| `wind_kw` | Nonnegative wind generation in kW |
| `load_kw` | Nonnegative demand in kW |

```bash
uv run microgrid prepare --csv /path/hourly.csv --output outputs/historical
uv run microgrid demo --csv /path/hourly.csv --output outputs/historical_demo
```

The pipeline rejects gaps, duplicate timestamps, nonfinite powers and negative powers. It writes Parquet and an audit report. Archive the source licence, measurement units and provenance yourself before research use. Raw irradiance and wind speed require the explicit conversion utilities in `src/microgrid/data.py`; the CSV command expects generation power already expressed in kW.

## Train and evaluate policies

After generating the default demo:

```bash
uv run --extra rl python scripts/rl_smoke.py --steps 1500
```

This trains and evaluates vanilla, fixed-reserve, adaptive-reserve and constrained SAC. Checkpoints and metrics are saved under `outputs/agents/`. These short runs verify integration, not convergence. See [EVALUATION.md](EVALUATION.md) for individual training and evaluation commands.

## Cloud and troubleshooting

The saved environment configuration contains a frozen dependency installation script and startup instructions. Review and save those settings, then publish the environment to retain the prepared snapshot. No live service needs restarting after restoration.

- If `uv` is missing, install it using its official instructions above.
- If dependencies are missing, rerun the frozen installation command with the required extras.
- If Matplotlib cache permissions are restricted outside the demo command, set `MPLCONFIGDIR=/tmp/microgrid-mpl` and `XDG_CACHE_HOME=/tmp/microgrid-cache`.
- An infeasible reserve is a reported model outcome, not an installation failure. Inspect `reserves.json` and the capacity assumptions.
- `outputs/` and `.venv/` are ignored by Git. Regenerate outputs after cloning; models and results are not included in the source repository.
