# Research protocol (proposal; empirical study not completed)

Primary question: at matched reliability, does causal joint-scenario adaptive reserve reduce battery damage relative to fixed reserve and strong conventional dispatch? Null: no benefit. Do not infer research novelty from implementing the pipeline.

Primary reliability outcome is EENS = Σ unserved_kW × Δt. Energy LPSP = EENS / total demand energy (defined as zero for zero demand). Report loss-of-load hours separately, not as LPSP. Provisional target is energy LPSP ≤ 0.01. Report fraction of physically infeasible scenarios, reserve availability, SOC trajectories, curtailment, throughput and computational latency. A reserve target is a soft operating preference, not permission to break physical bounds.

## Before an empirical experiment

1. Identify the public hourly dataset and archive original source, licence, timezone, measurement units, missing-data policy, site and capacity metadata. None was supplied in the development plan.
2. Audit hourly continuity; fit renewable sizing only on training weather using `fit_capacity_scaling`. Freeze conversion assumptions. Do not apply weather conversion to already-measured kW.
3. Freeze data hashes and chronological 60/15/10/15 split; tune on validation only. Freeze forecasts before calibrating completed joint residual blocks on separate calibration history.
4. Establish feasibility using perfect foresight, power adequacy, energy-deficit duration and size sweeps. If 100 kWh cannot meet the reliability target, report insufficiency instead of attributing deficits to policy weakness.
5. Replace illustrative ageing with a verified published LFP model. Record exact citation, parameter table, temperature applicability, cycle-depth/mean-SOC dependence, calendar assumptions, uncertainty and numerical reference validation. Current rainflow implementation validates cycle accounting only, not empirical battery lifetime.
6. Freeze controller configurations, final-test bounds, at least three seeds, stopping criteria, model hashes and all primary contrasts BEFORE final-test inspection. The demo manifest is a reproducibility artifact, not a completed preregistration.

## Planned matrix and statistics

Compare rule based, central-forecast MPC, fixed-reserve MPC/SAC, adaptive-reserve MPC/SAC, vanilla SAC, constrained SAC and two-stage scenario MPC. Perfect foresight is diagnostic only. Evaluate joint versus independent residual scenarios, with/without ageing reward, reserve observation versus penalty, generation-shortage stress, size/risk sweeps and random seeds. Use identical inputs and initial battery conditions. Include forecasting, sampling, reserve and controller latency in total runtime. Current demo reports controller-only latency and a separately labelled conservative shared preprocessing time.

Use paired circular chronological block-bootstrap confidence intervals (default 24 h blocks, 1000 replicates). The implemented interval is for mean hourly EENS difference; multiply by episode hours for total EENS difference. Choose block size using dependence diagnostics, examine sensitivity, and quantify variation across RL seeds separately. Compare degradation at matched reliability with Pareto fronts; raw damage reductions caused by shedding more load do not establish benefit.

## Current limitations

Synthetic weather/load are not a substitute for site measurements. Additive residual block bootstrap preserves historical joint dependence but does not guarantee tail calibration or capture regime shifts. P10/P90 interval coverage is measured but not corrected by conformal calibration. Quantile sorting removes crossing, not bias. Synthetic sizing is fixed for demonstration. Ageing is explicitly an illustrative sensitivity model; SOH does not evolve during dispatch. Temperature is fixed at an assumed 25°C for the unvalidated ageing placeholder. Scenario MPC has optimistic post-first-stage recourse. Short SAC smoke runs are not converged and supply no scientific evidence. No citation, superiority result or statistical significance is asserted.
