# Degradation-aware adaptive-reserve MPC: research protocol

This supersedes SAC as the primary research path. The hypothesis is **lower modelled battery damage at noninferior realized reliability**, not lower EENS than an EENS-optimal greedy controller. No IEEE-level novelty or field-lifetime claim is assumed.

## Data and holdout

- Training: 2022. The quantile forecaster predicts deviations from the known same-hour previous-week value, using causal lags and known Indian calendar/seasonal features.
- Joint residual library: completed out-of-training trajectories in 2023, conditioned on local origin hour and nearby day of year. These are overlapping trajectories across hours, not thousands of independent weather histories.
- Online calibration initialization: January–June 2024.
- Parameter and hardware selection: July–December 2024. All 2022–2024 data are development because previous results have already been inspected.
- Fresh holdout: January 1–June 24, 2025. No policy result from this period is used for selection.

The primary publisher workbook for 2024–2025 is SHA-256 verified against its Mendeley manifest and its 12,984 hourly demand readings exactly match the earlier mirror. DataCite independently confirms DOI `10.17632/y58jknpgs8.2`, the author list and CC BY 4.0. The workbook explicitly labels demand in MW. Its timestamps remain naive; Asia/Kolkata is a contextual interpretation corroborated by solar timing, not an explicit registry declaration. Earlier 2022–2023 demand remains a checksum-pinned hourly mirror; full raw reaggregation is not claimed. Original files, source hashes and verified registry metadata are retained locally and summarized in evidence.

This remains **scaled national demand paired with Jaisalmer weather**, not measured demand from a local campus or village. Scaling and generation capacities use 2022 only. Half-hour UTC/IST weather alignment assumes piecewise-constant hourly weather; causal guarantees apply to the processed power series, not independently validated field measurement availability. Wind conversion uses a simplified 10m power curve. These limits remain visible in the result, not silently treated as resolved.

## Calibration target and physics

The scenario generator samples synchronized whole residual blocks from the matching local origin hour and 90 nearest seasonal days. It uses a stable scale `max(training standard deviation, 0.25 × |forecast median|)`. It does not divide by vanishing quantile intervals. PV and wind are bounded by training-derived nameplate capacities, with PV set to zero for an interval wholly outside civil twilight according to NOAA solar geometry. Demand remains nonnegative without an invented upper cap.

The calibrated object is an upper envelope for **every contiguous DC-deficit interval in the next 23 hours**. A score measures the largest normalized underprediction across all intervals. Per-origin-hour score queues contain at most 90 completed historical days; a 24-hour delay ensures that every outcome used is strictly past. A finite-sample 95% score quantile widens the raw envelope. This targets empirical simultaneous path coverage; it does not make the bootstrap draws calibrated joint probabilities or establish exchangeable conformal guarantees under temporal drift.

The same forecast/scenario and calibrated-envelope inputs are available to the appropriate baselines. Record raw/corrected coverage by month and origin hour, envelope width, power/energy infeasibility, reserve violations and the count of historical blocks. A validation coverage gate of 90% is required before a positive calibration-supported result is considered. The fresh holdout independently tests this gate. Infeasible reserve intervals are retained and request full physical storage; they are not conditioned away.

## Controllers and ageing objective

Every operational controller serves all physically feasible measured current demand before optimizing surplus charging. This prevents the earlier failure mode of withholding available energy from present load to satisfy forecast preferences.

MPC uses lexicographic objectives:

1. Minimize forecast expected EENS with a common first action across scenarios.
2. Minimize reachable reserve violations within the selected forecast-EENS budget.
3. Minimize a published-model-derived wear surrogate at the same preceding bounds.

The surrogate combines calendar SOC exposure and DC throughput using marginal coefficients from the BLAST-Lite 250Ah LFP-Gr model at declared reference age/EFC. It is a computational proxy, not the exact offline degradation functional. Future scenario reserve constraints are on expected stored energy; only the first energy state is common. The scenario optimizer is an optimistic two-stage benchmark, not a fully non-anticipative multistage tree.

Baselines include greedy, deterministic degradation-aware MPC, validation-tuned constant reserve, stochastic degradation-aware MPC with tuned expected-EENS slack, and stochastic MPC with the same envelope. Weekly-persistence point and separately tuned fixed-reserve controls are included because persistence outperformed the learned forecast on development MAE. An independently sampled residual method receives its own calibration and equally refined safety buffer. The best validation-qualified baseline by published-model damage is frozen as the primary comparator. Proposed safety buffers are selected from a declared grid including a full-reserve fallback, then fixed/adaptive/independent parameters receive equal 2 kWh search precision. Increasing a safety buffer cannot be presented as improved calibration. Raw-envelope and no-calendar-cost ablations use the proposed method's selected safety buffer, avoiding a changed-buffer confound.

Selection minimizes modelled damage only among validation-reliability-eligible candidates. Unqualified methods remain reported and cannot support a matched-reliability improvement claim. Fixed-reserve and stochastic risk settings receive validation tuning as well. Search expansions prompted by development failures are recorded; no holdout outcomes inform them.

## Matched reliability and terminal inventory

The EENS noninferiority margin is 0.0001 × operational demand energy: **0.01 percentage points of energy LPSP**. Report point differences and paired block-bootstrap upper bounds for 24-hour and 168-hour blocks. A positive ageing claim requires both reliability bounds below the margin, plus a negative upper bound on paired model-damage differences.

The final 24 hours use one common **recharge-only normalization policy** for every method: renewable surplus may charge, current renewable generation serves demand directly, and battery discharge is disabled. There is no grid import, invented energy or future-information oracle. The day normalizes terminal energy to the physical maximum when resources permit. Its unserved load is reported separately from operational EENS, while ageing includes the normalization period. If a method cannot attain the common final inventory, endpoint matching fails and no endpoint-matched damage benefit is claimed. This is an explicit experimental boundary condition, not a proposed operational rule for a deployed microgrid.

## Battery-model verification and scope

Historical upstream-Euler integration remains available so earlier results are reproducible. The revised evaluator also provides an exact effective-age power-law step, checked against constant-stress analytic solutions and time-step composition. Rates and parameters remain tied to the pinned published reference; no battery-life coefficient is fitted to favor a controller.

Offline evaluation covers assumed battery temperatures 25°C and 35°C, and update windows 12/24/48 hours. Report calendar and cycling components, equivalent full cycles, throughput and the fraction of windows outside the published DOD range. Numerical agreement with reference code, analytic integration tests and sensitivity stability are separate from empirical pack validation. No local battery test data are available; no validated lifetime-extension percentage is claimed. Controlled battery temperature is an assumption and cooling energy is not modelled.

## Reproduction and release

`configs/age_mpc/protocol.yaml` records settings; the script implements them explicitly rather than loading arbitrary YAML. `scripts/age_mpc_study.py` stages `prepare`, `select`, `refine`, `test`, `report` preserve development caches and freeze data/code hashes before the fresh holdout. `seal_age_mpc_run.py` completes pre-test artifact receipts. Frozen results are never silently overwritten. Original synthetic and prior Indian-reference results remain separate. Full hourly traces, inputs, adaptive current reserve requests, runtime and normalization flags accompany aggregate results. Fixed-controller requested-reserve trace fields are defaults rather than a stored fixed-reserve trajectory; the fixed fraction is recorded in the frozen candidate metadata.

Frozen policies also receive daily capacity-fade feedback sensitivities at assumed 25/35°C, preserving SOC and explicitly accounting for lost stored energy. These are not additional policy-selection opportunities. The serialization-only wrapper `run_age_mpc_coupled.py` preserves the primary numerical source. See [reproduction](age_mpc_reproduction.md) and [completed findings](../MPC_RESEARCH.md).

Further SAC training is paused. If reliability, calibration, endpoint or battery-applicability gates fail, report a negative or inconclusive result and the precise failed gate. Novelty review and independent application validation remain prerequisites to an IEEE submission.
