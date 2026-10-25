# Next experiment: India-based feasibility, calibration and reserve usefulness

Status: proposed follow-up, not an executed empirical study. Preserve the October 24 synthetic baseline and its recorded negative result. Do not increase SAC training budgets until the following gates have been assessed.

## 1. Data selection and provenance

The selected geography is **India**. Use a **Jaisalmer, Rajasthan** reference site (approximately 26.9157° N, 70.9083° E) as a proposed solar–wind case study, subject to data availability. This is a geographic reference, not an identified operating microgrid. Follow [India case-study data plan](india_case_study.md) for source selection, load provenance, UTC/IST alignment and acceptance gates. Prefer at least three continuous years of aligned hourly weather and load, with seasonal coverage and explicit usage and redistribution terms. Record source URLs, retrieval dates, raw-file hashes, site coordinates, timezone/DST policy, irradiance/wind measurement heights, load units and capacity assumptions. Public availability alone does not establish permission to redistribute.

If weather and demand come from different locations or organizations, identify the case as a constructed microgrid scenario rather than a measured co-located microgrid. Do not treat a weather API's derived PV estimate as measured generation. Convert raw weather through documented models. Estimate any renewable sizing parameters using training data only.

Retain the 60/15/10/15 chronological split. Audit all timestamps, gaps, unit conversions and boundary conditions before training. Choose and record any gap-repair policy in advance; do not silently fill prolonged renewable shortages.

## 2. Physical adequacy gate

Use training and validation to study battery energy and power separately, rather than treating nominal kWh as sufficient. Candidate energy capacities: 100, 200, 400 kWh; power limits: 25, 50, 100 kW; renewable scale factors relative to a training-derived reference: 0.75, 1.0, 1.25, 1.5. These are exploration settings, not a claim that any design is feasible.

Compare rule based and complete-episode perfect foresight for each design, with identical initial SOC and boundary assumptions. Include sensitivity to initial SOC and terminal energy, and use continuous episodes rather than resetting to free energy every day. Separate instantaneous power shortage, cumulative energy shortage, insufficient generation and storage timing. Report renewable/demand energy, peak deficits, EENS, energy LPSP, curtailment, minimum-SOC hours and terminal energy.

The provisional target remains 1% energy LPSP. If the perfect-foresight diagnostic cannot achieve it, record inadequacy and investigate sizing before controller tuning. Passing this diagnostic only establishes possibility under future knowledge, not operational reliability. Select the system design on validation and freeze it before final-test inspection.

## 3. Forecast and scenario calibration gate

Keep raw quantile forecasts as a reference. Report MAE, RMSE, pinball loss, coverage and interval width separately for PV, wind and load, at every lead from 1 to 24 hours. Separate daylight PV coverage from night zeros and inspect shortage seasons and sustained errors.

Evaluate a causal quantile-interval correction on held-out calibration observations. Predeclare a division of the existing calibration period into interval-correction and residual-scenario fitting windows, or use another documented nested chronological design. Evaluate the correction only on later data. Do not fit a correction on the same outcomes used to claim improved coverage. Temporal dependence limits simple exchangeable conformal guarantees; report empirical coverage and assumptions rather than claiming automatic guarantees.

Keep synchronized 24-hour residual trajectories. Record the number of unique completed historical blocks, unique sampled blocks and effective seasonal coverage; drawing 100 scenarios does not create 100 independent weather histories. Compare joint and independent residual scenarios on unseen cumulative net-load deficits, cross-source dependence, temporal autocorrelation and prolonged shortage duration. Calibrating marginal P10/P90 intervals alone does not calibrate the P50 residual bootstrap or its reserve tail.

## 4. Reserve go/no-go experiment

On validation only, compare rule based, deterministic MPC, fixed-reserve MPC, adaptive-reserve MPC and scenario MPC using identical measured first-stage inputs. Tune the fixed reserve on validation; comparing only against an arbitrary 40% target is insufficient. Predeclare candidate reserve fractions and risk levels (for example fixed fractions 0.2–0.8 of effective capacity, alpha 0.01/0.05/0.1) and preserve all infeasible outcomes. Do not choose risk solely to eliminate infeasibility.

Run diagnostic counterfactuals: no reserve penalty, fixed versus adaptive time-indexed targets, marked infeasibility fallbacks, and actual future inputs solely in a separately labelled non-deployable diagnostic. Inspect hours where load is shed despite energy above the physical floor. Such hours may reflect scheduling choices, future deficits, power limits or forecast error; the SOC indicator alone cannot identify the cause.

Report initial reserve, suffix reserve targets, infeasible scenario mass, power versus capacity infeasibility, unmet reserve, and actual dispatch outcomes. Compare throughput at matched reliability; otherwise reduced battery use can simply mean more unserved demand. If no useful reserve benefit emerges, retain the negative result and revise the hypothesis before adding learning complexity.

## 5. Lifecycle and learning gate

Use throughput and rainflow equivalent full cycles as separate operational diagnostics until a suitable published LFP model has been selected, cited, parameterized and checked against independent reference calculations. Do not infer lifetime extension from the current illustrative parameters. Explicitly decide whether SOH evolves within dispatch or is a fixed scenario input.

Only after the earlier gates, select increasing SAC budgets on validation, use multiple seeds, inspect convergence and reliability compliance, and freeze checkpoints and hyperparameters before the final test. Include full forecast/scenario/reserve/controller runtime and paired chronological uncertainty. A learned constraint penalty does not guarantee compliance.

## Available baseline review package

The current cloud workspace contains `outputs/audit/baseline_review.zip`, including requested metrics, forecast validation, reserve diagnostics, paired bootstrap results, all seven dispatch Parquet traces, generated hourly input data, a plot, manifest and SHA-256 checksums. It also contains hourly adaptive-versus-rule-based comparisons and controller diagnostics. The archive was checked for ZIP integrity. Generated artifacts remain ignored by Git; the archive has not been uploaded to an external service.

The diagnostic identifies adaptive MPC unserved load of 118.265 kWh in hours ending above minimum battery energy, compared with zero for rule-based control. This is an investigation flag, not proof those deficits were all avoidable over the full horizon. All checked AC balance residuals were below 4e-15 kW. The original 72-hour EENS results remain unchanged.
