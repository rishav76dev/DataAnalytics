# Reproduce the constructed Indian reference experiment

This workflow executes research comparisons, rather than merely planning them. Read [INDIA_RESULTS.md](../INDIA_RESULTS.md) for outcomes and source limitations. It uses national aggregate demand scaled to a reference load, not a measured village microgrid. The primary demand record remains inaccessible through the current proxy; the pinned secondary mirror documents a CC BY 4.0 dataset and original DOI. Do not present secondary-source assertions as independently verified primary metadata.

## Prerequisites and sources

From the repository root, install Python dependencies:

```bash
uv sync --frozen --extra dev --extra rl
```

CPU-only PyTorch is used. The executed environment had four CPU cores available and about 32 GiB free RAM. A complete conventional run took minutes; the 12 SAC runs take substantially longer. No GPU or API key is required. NumPy/SciPy implement the published-model evaluator directly, so no conflicting BLAST-Lite package environment is installed. BSD notices and pinned reference excerpts are retained under `src/microgrid/BLAST_LICENSE.txt` and `tests/reference_blast/`.

Weather: NASA POWER, Jaisalmer (26.9157 N, 70.9083 E), hourly UTC irradiance, temperature and 10m wind. Demand: `RudranshVyas/india-grid-forecast`, pinned commit `9fc756aeceefb2e53d6494d48722f5dd771996e0`, `data/processed/hourly_raw.parquet`, SHA-256 `c77ed3e200565c5609bd52806bef98f4e7eac22da6609d60750c8ce7a5fb75ed`. The mirror attributes the measurements to Kumar, Mukherjee and Kalita (2025), Mendeley Data V2, DOI `10.17632/y58jknpgs8.2`. Its units, timezone and licence statements remain secondary-source evidence pending primary verification.

The importer checks this exact pinned-file checksum. It retains 2022–2024 demand, causally fills six implausible points with a maximum three-hour fill, scales MW to a training-mean 12 kW proxy, and aligns IST hourly demand with two adjacent UTC weather hours using equal overlap. No source-specific historical observations are invented. The reference generation model remains simplified, especially its use of 10m wind without a validated hub-height conversion.

## 1. Acquire and prepare

```bash
uv run python scripts/india_research.py fetch-weather \
  --start-year 2020 --end-year 2024 --output data/india/weather
uv run python scripts/prepare_india_reference.py
```

The first command saves original annual responses and checks full-year coverage. The second downloads the pinned demand mirror if absent, verifies its hash, and writes `data/india/reference/hourly.parquet` and `audit.json`. The constructed input contains 26,304 hours. The audit records transformations and provenance limits explicitly. These data files remain ignored by Git.

## 2. Validation adequacy and boundary sensitivities

```bash
uv run python scripts/india_research.py feasibility \
  --data data/india/reference/hourly.parquet --data-status constructed_india \
  --output outputs/india_study/sizing \
  --capacities 100 200 400 --powers 25 50 --renewable-scales 1 1.25 \
  --initial-socs 0.1 0.5 --terminals minimum initial

uv run python scripts/india_research.py feasibility \
  --data data/india/reference/hourly.parquet --data-status constructed_india \
  --output outputs/india_study/sizing_expanded \
  --capacities 200 400 800 1600 --powers 50 --renewable-scales 1.5 2 3 \
  --initial-socs 0.5 --terminals minimum initial
```

The second grid was explored after all initial designs failed the provisional 1% validation energy-LPSP target. No test data are used for sizing. These are 72 configuration/boundary cases, not 72 independent datasets. The final design selection minimizes capacity, then renewable scale, then power among validation-adequate cases with initial SOC 0.5 and final energy at least initial energy. It is not a capital-cost optimum.

## 3. Freeze the conventional comparison and evaluate once

```bash
uv run python scripts/final_india_benchmark.py
```

This trains the central forecaster on outer training, selects a fixed reserve using validation, fits 109 completed joint residual blocks on separate calibration, and writes code/data/model hashes before final dispatch. It evaluates the **entire 3,947-hour test partition**, seven controllers and a complete-episode EENS-optimal perfect-foresight reference. It also calculates published-model offline ageing and records end inventory and runtime scope.

The script refuses to overwrite an existing `frozen_manifest.json`. For a deliberate replication use a new `--output` directory and label it as a replication rather than silently replacing the first test. Existing outputs do not establish correctness; inspect completed metrics and traces. The original frozen source is archived because later independent helper additions may change the current source tree. Its per-file hashes match the pre-test manifest.

## 4. Three-seed SAC budget comparisons

Once the conventional manifest exists (the conventional run may still be dispatching), training can begin:

```bash
uv run --extra rl python scripts/india_rl_study.py train
uv run --extra rl python scripts/india_rl_study.py evaluate
```

Defaults are four variants, seeds 42/43/44, and 10,000/30,000-step candidates, totaling 360,000 training steps. The second budget continues the same model rather than retraining from scratch. Validation checkpoint selection minimizes EENS, then daily published-model damage. Selection and checkpoint hashes are saved before any RL final-test evaluation.

A nested causal warmup supplies training reserve observations without fitting a forecaster on future policy-training labels. All variants use the same final suffix of outer training and identical validation/test episodes. This suffix is shorter than a full year; do not claim full seasonal policy training. The shared published-model damage reward is an hourly proxy, evaluated separately against a daily offline calculation. Twelve observations include ageing states. Weekly training episodes restart initial SOC consistently; final evaluation remains continuous without daily resets.

The constrained variant applies a nonnegative episode-wise dual multiplier. The reliability target can remain unmet; in fact the fixed-capacity perfect-foresight result diagnoses an out-of-sample design shortfall. Learning cannot remove that physical lower bound. Budget comparisons and three seeds quantify variation, not guaranteed convergence. These comparisons do not constitute every ablation in the original proposed research matrix.

## 5. Analysis and evidence export

```bash
uv run python scripts/analyze_india_study.py
uv run python scripts/export_india_report.py
```

Analysis reuses trusted local forecaster checkpoints; never load unknown pickles. It reports pre-specified interval correction using early validation only, evaluates scenario net-energy tails, compares 24-hour and 168-hour paired bootstrap intervals, and produces a reliability–modelled-damage plot. The matched-reliability screen is descriptive and explicitly reports its tolerance; it does not select another policy on the final test.

The export saves summary JSON/CSV, chosen checkpoints' metadata, source hashes, plots and frozen source under `docs/evidence/india_reference_2026-10-25/`. It updates `INDIA_RESULTS.md` and the manuscript assets. Full traces and large models remain under ignored `outputs/`.

## 6. Ageing and physical feedback

`microgrid.ageing` implements the pinned BLAST-Lite LFP-Gr 250Ah parameterization and compares its coefficients/state updates against a source reference in tests. The upstream model cites experimental DOI `10.1016/j.est.2023.109042`. Temperature and rate envelopes are checked; cycle-depth extrapolation is reported. Source-code agreement is not empirical pack validation.

`microgrid.ageing_dispatch.rule_based_with_capacity_fade` provides a separately labelled daily capacity-feedback sensitivity. SOC is preserved as capacity shrinks, removed stored energy is recorded, and subsequent actions obey the updated physical limits. The primary comparison retains fixed capacity for all methods; the RL reward's internal ageing proxy does not silently alter those primary battery limits. The recorded 25°C and 35°C sensitivity is post-hoc, not a newly tuned policy.

## 7. Verification and report build

```bash
uv run --extra dev --extra rl pytest -q
uv run --extra dev ruff check src scripts tests
```

Thirty tests passed in the executed environment, including independent LP comparisons and pinned ageing-reference regression. Reference third-party files are excluded from automatic formatting to retain their documented upstream content; their tests execute normally.

For the experimental manuscript, install a standard TeX distribution if needed, then from `paper/` run `pdflatex manuscript.tex`, `bibtex manuscript`, followed by two more `pdflatex manuscript.tex` passes. Keep generated auxiliaries in a local output directory. The executed cloud build uses `/tmp/microgrid-paper` and copies the final PDF into the evidence directory after compilation. Compilation is not a claim that the report is ready for IEEE submission.

## Research gate

Read [greedy_reliability_limit.md](greedy_reliability_limit.md) before further SAC budget escalation. Under the implemented EENS-only assumptions, greedy is already optimal; the observed equality with perfect foresight is therefore structurally expected. An adaptive ageing advantage must be established at matched reliability, with validated application physics and source provenance. The existing experiment does not establish that advantage.
