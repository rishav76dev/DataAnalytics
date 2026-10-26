# Frozen degradation-MPC evidence

Read [results](../../../MPC_RESEARCH.md), [protocol](../../degradation_mpc_protocol.md), [validation](../../age_mpc_validation.md) and [reproduction](../../age_mpc_reproduction.md). This is a constructed India reference experiment, not a measured local microgrid or empirical battery-life validation.

- `controller_results.csv`, `test_metrics.json`: all eleven frozen controllers, including methods that fail reliability matching. Operational EENS excludes the common final 24-hour normalization day; total EENS is also provided in the CSV.
- `trace_<controller>.parquet`: **full 4,200-hour primary traces**, with UTC timestamps, bus powers, DC battery energy, SOC, reserve requests/infeasibility, damage increments and normalization flags. These files are committed rather than left only in ignored output directories.
- `trace_coupled_*.parquet`, `coupled_metrics.json`: six complete frozen-policy capacity-feedback sensitivities at 25/35°C. Physical SOH changes daily, with lost stored energy explicitly accounted for.
- `matched_reliability.json`, `claim_gates.json`: paired daily/weekly intervals and the failed superiority decision. The best validation-qualified independent reserve remains the primary comparator even though it fails test reliability matching.
- `ageing_sensitivity.csv`, `surrogate_diagnostic.*`: offline temperature/update-window sensitivity and a post-test proxy-ranking diagnostic. These do not constitute empirical battery validation.
- `test_calibration*.json`, `test_independent_calibration.json`: aggregate and full per-origin path-coverage records. Nominal 95% coverage is not attained throughout the holdout.
- `forecast_future_diagnostics.json`: per-variable/per-lead future forecast metrics; fixed marginal corrections are diagnostic only. The older development report includes lead zero and uses a different averaging population.
- `data_audit.json`, `datacite.json`, `files_manifest.json`, `weather*_manifest.json`: primary-source attribution, rights, units, transformations and hashes. Demand is CC BY 4.0; retain the attribution below when reusing derived traces.
- `frozen_protocol.json`, `freeze_receipts.json`, `frozen_source.zip`: parameter/data/model/code fingerprints and the actual numerical source before test evaluation. The original cloud script path appears absolute in the receipt; the verifier resolves it portably. `test_evaluated: false` records the state **when frozen**, not the current completion status.
- `coupled_serialization_receipt.json`, `coupled_runner_at_execution.py`: disclosure and exact executed wrapper source for a NumPy-boolean serialization repair, which changed no numerical code or policy parameters. The repository wrapper later received import formatting; the receipt hash identifies the archived execution version.
- `environment_versions.json`, `trace_verification.json`, `checksums.json`: dependency versions, full-trace physical ledger checks and release file integrity.

Verify from the repository root:

```bash
.venv/bin/python scripts/verify_age_mpc_release.py
```

Derived demand attribution: Debanjan Mukherjee, Karuna Kalita and Subhash Kumar (2025), Mendeley Data V2, [10.17632/y58jknpgs8.2](https://doi.org/10.17632/y58jknpgs8.2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The study scales national demand, repairs six earlier hours and combines it with converted Jaisalmer NASA POWER weather. It is not endorsed by the source authors. NASA POWER project data are acknowledged. Model-source BSD terms are retained inside the source archive and [repository licence notice](../../../src/microgrid/BLAST_LICENSE.txt).

Large raw annual responses, scenario arrays and local model pickles are deliberately not required to review the release. The exporter also creates `outputs/age_mpc_v2/review_bundle.zip` locally, containing this evidence directory.
