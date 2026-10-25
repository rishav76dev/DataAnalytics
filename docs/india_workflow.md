# Executable India research workflow

Historical workflow note: a later national-demand-mirror reference experiment is now executed. See [india_research_execution.md](india_research_execution.md) and [../INDIA_RESULTS.md](../INDIA_RESULTS.md) for its current status. The commands below retain the stricter path for independently verified local-demand metadata.

Implementation status: acquisition, alignment, adequacy, interval-correction and reserve-validation commands are implemented. Five complete years of Jaisalmer weather (2020–2024, 43,848 hourly samples) have now been downloaded and audited. An Indian hourly demand series is still missing, so no Indian dispatch results have been generated. Existing synthetic results remain unchanged. Published LFP model validation and extended SAC experiments remain gated on evidence, not represented as complete.

Run from the repository root after the installation in [SETUP.md](../SETUP.md). No new dependencies beyond the frozen environment are required.

## 1. Download Jaisalmer weather

```bash
uv run python scripts/india_research.py fetch-weather \
  --start-year 2020 --end-year 2024 --output data/india/weather
```

Requests one complete year at a time from NASA POWER with UTC explicitly requested. Stores original JSON, SHA-256 hashes, request URLs, site metadata, unit-conversion notes and continuous hourly weather Parquet. Existing raw files are reused only after parsing and checking coverage. Missing values, unexpected units, local solar time and incomplete years fail rather than becoming usable data silently. TLS verification remains enabled. The parser has fixture tests and successfully parsed all five live annual API responses.

The initial connection attempt returned proxy CONNECT HTTP 403; subsequent access succeeded after the domain update. A configuration draft adds `power.larc.nasa.gov`, `grid-india.in`, and `cea.nic.in` while preserving package-manager presets. Review and save the draft in environment settings, then publish it; a draft save is not itself proof of runtime access; the later successful download establishes current-instance NASA access. No new secret is required for the intended NASA request. Additional hosts discovered during official source navigation would need separate justified additions.

## 2. Obtain and prepare demand

A defensible Indian hourly demand source is still required. Follow [india_case_study.md](india_case_study.md): a state aggregate profile is not a measured community microgrid. Do not substitute synthetic demand while labelling the case historical.

Create `timestamp,load_kw` CSV with explicit timezone offsets and interval-start conventions. Copy `configs/india/load_metadata.example.json` to a local file and replace every placeholder with verified source facts; set permission true only when reuse is actually permitted by the recorded terms. The importer rejects false reuse permission and placeholder metadata. Metadata assertions are user-supplied provenance, not independent licence verification.

```bash
uv run python scripts/india_research.py prepare \
  --weather data/india/weather/weather.parquet \
  --load /path/indian_hourly_demand.csv \
  --metadata /path/verified_load_metadata.json \
  --output data/india/processed
```

The importer takes the common weather/demand time range, then requires exact, continuous timestamp alignment within it. It does not round half-hour offsets or silently discard interior gaps. Convert IST interval integration windows correctly before this command; timezone conversion alone does not align IST clock-hour windows with UTC clock hours. It fits PV/wind capacities on the training portion only and writes hourly CSV/Parquet and an audit with source hashes. The simplified wind model uses the supplied 10m input; the explicit height assumption is required and a validated turbine/hub-height model remains research work.

## 3. Adequacy and sizing on validation only

```bash
uv run python scripts/india_research.py feasibility \
  --data data/india/processed/hourly.parquet --data-status constructed_india \
  --capacities 100 200 400 --powers 25 50 100 \
  --renewable-scales 0.75 1 1.25 1.5 \
  --initial-socs 0.1 0.5 --terminals minimum initial \
  --output outputs/india/feasibility
```

This study optimizes the complete validation episode once per design using a sparse LP whose sole objective is EENS. It reports a non-deployable optimum, rule-based EENS, an unavoidable power-shortage lower bound, renewable energy, end energy, and balance residual. No daily free-energy resets are inserted. `initial` terminal policy requires final energy at least the episode's starting energy. Report each policy separately; the accompanying rule-based result does not enforce the oracle's optional terminal-energy requirement, so it is not a matched-terminal ranking.

The `--data-status` value is a user declaration preserved in the manifest, not automatic provenance certification. Test outcomes are not used for sizing selection.

## 4. Held-out interval diagnostics

```bash
uv run python scripts/india_research.py calibration \
  --data data/india/processed/hourly.parquet --data-status constructed_india \
  --output outputs/india/calibration
```

Fits the forecaster on training, interval widening on the first half of validation, and evaluates on the later half of validation. Corrections are specific to variable and horizon, leave P50 unchanged and never shrink intervals. Separate outer calibration data supplies completed joint residual blocks. Reports raw/corrected coverage, width and errors by variable/horizon, plus observed-nonzero PV diagnostics. There is no exchangeable-coverage guarantee for dependent hourly data. Widening marginal intervals does not modify the median-residual sampler or establish its shortage-tail calibration.

## 5. Tune fixed reserve before comparison

```bash
uv run python scripts/india_research.py reserve \
  --data data/india/processed/hourly.parquet --data-status constructed_india \
  --hours 72 --output outputs/india/reserve
```

Forecasts are trained on the first 80% of the outer training partition; completed residual blocks come from its remaining 20%, preceding all validation outcomes. A fixed fraction is selected from 0.2/0.4/0.6/0.8 on the first validation half, minimizing EENS then throughput. Selection is written before evaluating later validation hours. Default episodes use 72 hours from each half; this is a diagnostic budget, not a full seasonal study.

Later validation compares rule based, no-reserve deterministic MPC, tuned fixed reserve, adaptive reserve with maximum-energy fallback, adaptive reserve with infeasible targets removed (physical-minimum soft target), scenario MPC and an EENS-optimal perfect-foresight diagnostic. All physical constraints remain hard. Saves hourly traces, initial/suffix infeasibility, paired bootstrap intervals and unserved energy occurring above the battery floor. Removing an infeasible target does not make its scenario feasible. No final test is evaluated by these commands.

The sizing, calibration and reserve commands are separate diagnostics. The reserve command currently uses the baseline 100 kWh/50 kW battery and nested historical forecaster; it does not automatically ingest a sizing-study winner or interval-correction artifact. Freeze and explicitly implement the selected experimental configuration before any empirical final test. Do not present these independent checks as a completed end-to-end calibrated Indian controller.

## Evidence and outstanding requirements

See [research_followup_results.md](research_followup_results.md) for actual executed synthetic checks. Still needed: an audited Indian demand series; historical aligned experiments and shortage-tail diagnostics; a cited, independently validated LFP model; validated training budgets and multiple SAC seeds. No lifecycle-extension or adaptive-superiority claim is supported yet.
