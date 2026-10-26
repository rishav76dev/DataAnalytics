# Reproduce the degradation-aware MPC study

Read [the frozen results](../MPC_RESEARCH.md) and [protocol](degradation_mpc_protocol.md) first. SAC research training is paused. This workflow consumes CPU and disk; it is not a startup command. The release contains the full 2025 traces so inspecting the findings requires no downloads or training.

## Inspect the committed release

From the repository root, using Python 3.12 and the pinned environment:

```bash
export UV_CACHE_DIR=/tmp/uv-cache
export UV_LINK_MODE=copy
uv sync --frozen --extra dev --extra rl
.venv/bin/python scripts/verify_age_mpc_release.py
.venv/bin/pytest -q
.venv/bin/ruff check src tests scripts
```

`docs/evidence/age_mpc_v2/` includes all eleven primary traces and six capacity-feedback traces, aggregate metrics, source metadata, calibration records and the checksum inventory. The verifier reads these artifacts without loading a pickle, downloading data or rerunning a policy. `frozen_source.zip` is the actual pre-test simulation source and protocol, including the BLAST licence. Its archive hash and protocol hash are recorded in the pre-test receipt. A receipt shows local experiment ordering; it is not externally timestamped preregistration.

## Recompute in a clean checkout

Keep the released evidence. Use a fresh clone/checkout with empty `data/india/age_mpc_v2/` and `outputs/age_mpc_v2/`; do not delete a completed experiment just to rerun it. All commands use paths relative to the repository root.

```bash
.venv/bin/python scripts/prepare_age_mpc_data.py
.venv/bin/python scripts/age_mpc_study.py prepare
.venv/bin/python scripts/age_mpc_study.py select
.venv/bin/python scripts/age_mpc_study.py refine
.venv/bin/python scripts/seal_age_mpc_run.py
.venv/bin/python scripts/age_mpc_study.py test
.venv/bin/python scripts/run_age_mpc_coupled.py
.venv/bin/python scripts/age_mpc_forecast_audit.py
MPLCONFIGDIR=/tmp/microgrid-mpl XDG_CACHE_HOME=/tmp/microgrid-cache \
  .venv/bin/python scripts/export_age_mpc_evidence.py
.venv/bin/python scripts/verify_age_mpc_release.py
```

The data command caches complete NASA POWER annual UTC responses for 2020–2025, downloads the pinned hourly demand mirror, verifies the publisher workbook checksum/licence, checks exact primary overlap, and builds 30,504 hours for 2022–June 2025. The extra earlier weather supports local-time boundary alignment. Network access is needed for NASA POWER, GitHub, DataCite and Mendeley including its download redirect. Upstream NASA revisions can change the processed-data hash; compare manifests rather than silently accepting a different dataset as exact reproduction.

`select` performs the coarse hardware and parameter study on development data. `refine` gives fixed, adaptive and independent reserve parameters the same 2 kWh search precision and includes tuned weekly-persistence controls. It freezes the best validation-qualified baseline as the primary comparator. `seal` adds artifact hashes and a source archive without changing selections. Test checks frozen data/code/model hashes and refuses a completed holdout. Do not change frozen numerical files after sealing. The independent calibration state is separately fitted and hashed.

`configs/age_mpc/protocol.yaml` is the recorded specification; this study script implements those settings explicitly. It is not an arbitrary YAML configuration loader. Random seeds are fixed; exact runtime and compressed/serialized bytes can vary with hardware and library build. Check numerical results, source hashes, primary-source hashes and dependency versions separately. Large scenario arrays, trusted local model caches and raw data remain ignored. No untrusted serialized model is required to inspect the release.

The original coupled run exposed a NumPy-boolean JSON serialization error after calculation and replay checks. `run_age_mpc_coupled.py` converts NumPy scalars only and preserves the frozen simulation source byte-for-byte. Use this wrapper for the capacity-feedback sensitivity. It refuses an already completed coupled study. This is a disclosed serialization repair, not a test-informed policy change.

The additional forecast audit computes marginal diagnostics from a January–June 2024 fixed correction, separately by variable and future lead 1–23. These diagnostic intervals do not change the controller envelope or actions. The originally archived development marginal report includes the current lead; the new `forecast_future_diagnostics.json` excludes it and is the preferred future-error diagnostic. Results therefore differ slightly between the two reports.

## Read hourly traces

```python
import pandas as pd
trace = pd.read_parquet(
    "docs/evidence/age_mpc_v2/trace_adaptive_age.parquet"
)
operational = trace.loc[~trace.normalization]
print(operational.unserved_kw.sum())  # hourly steps: kW × 1 h = kWh
```

Timestamps are UTC; convert to Asia/Kolkata for local plots. `battery_kw` is positive AC discharge and negative charging. `energy_kwh` is the post-step DC energy. `throughput_kwh` is DC energy throughput. `requested_reserve_kwh` is the adaptive current end-of-step request; its default value for fixed/rule controllers is not a logged fixed-reserve trajectory. `reserve_infeasible` flags the adaptive raw physical requirement rather than hiding infeasible scenarios. No artificial feasibility claim follows from capping the operational fallback at full storage.

`damage_*` columns contain model-loss increments at update boundaries, not continuously measured hourly wear. `physical_soh` is one in the primary fixed-capacity experiment and changes daily only in coupled sensitivities. `capacity_fade_energy_loss_kwh` records the energy removed when capacity fades at constant SOC. Coupled terminal matching is against each battery's remaining maximum capacity; use `terminal_full_current_capacity`, not the primary fixed-capacity terminal flag.

The common final recharge-only day forbids discharge and can shed demand even when stored energy is available. It gives equal terminal inventory using real renewable charging. Always report normalization EENS alongside operational EENS, and include that day in ageing. This boundary treatment limits direct deployment interpretation.
