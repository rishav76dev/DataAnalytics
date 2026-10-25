# Follow-up implementation and executed evidence

Executed on 2026-10-25. These are new validation diagnostics; they do not replace or revise the October 24 synthetic final-test summary in RESULTS.md.

## Indian data acquired

NASA POWER weather was downloaded for Jaisalmer, Rajasthan (26.9157 N, 70.9083 E), 2020-01-01 00:00 UTC through 2024-12-31 23:00 UTC: **43,848 hourly rows**, with no missing hours or rejected missing values. Parameters: hourly surface irradiance (Wh/m² converted to hourly mean W/m²), 2m air temperature (°C), and 10m wind (m/s). These are gridded weather estimates, not measured plant output or community demand.

Original annual responses and processed weather are retained in ignored data/india/weather/. Source URLs, exact raw hashes and audit summary are versioned in [evidence/2026-10-25](evidence/2026-10-25). NASA documentation and acknowledgement pages were consulted; no third-party raw dataset is redistributed in Git. Usage/redistribution terms must be archived before redistribution. CEA homepage was reachable but did not establish an hourly community-demand dataset; Grid India returned a connection failure. A GitHub mirror was also inspected: [India-Hourly-Energy-Consumption-Forecasting](https://github.com/MehulPrajapati1/India-Hourly-Energy-Consumption-Forecasting), file `hourlyLoadDataIndia.csv`. Its README claims national hourly demand, but the inspected file has naive timestamps and a `National Hourly Demand` column without explicit units. The inspected README does not establish original provenance, timezone or data reuse terms. It was not adopted or redistributed; national demand would in any case be a constructed profile proxy, not community measurements.

## Software verification

- 22 automated tests passed after adding India parsing/alignment, interval correction and adequacy tests; existing Gymnasium advisory warnings remain.
- Ruff lint and formatting passed.
- Live weather import completed for all five annual responses; UTC declarations, units and yearly continuity checked.
- 72 synthetic validation sizing designs executed, varying capacity, power, renewable scale, initial SOC and terminal policy.
- Calibration and reserve-counterfactual CLIs executed on synthetic validation only. No further SAC training was run.

## Adequacy diagnostic on synthetic validation

| Nominal capacity | Power | Renewable scale | Initial SOC | Terminal policy | Perfect-foresight energy LPSP |
|---|---|---|---|---|---|
| 100 kWh | 50 kW | 1.0 | 0.5 | Physical minimum | 14.47% |
| 200 kWh | 50 kW | 1.0 | 0.5 | Physical minimum | 0.75% |
| 400 kWh | 50 kW | 1.0 | 0.5 | Physical minimum | 0.00% |

32 of the 72 explored designs achieved <=1% energy LPSP under perfect foresight on this synthetic validation partition. This characterizes these inputs and episode boundaries only, not Indian equipment adequacy or deployable reliability. Full results and terminal-policy sensitivities are in the evidence CSV.

## Held-out interval correction on synthetic validation

Raw coverage: **58.80%**; corrected coverage: **78.47%**, against a nominal 80% interval.

Correction fitting used 6 completed daily origins in early validation; evaluation used 6 later origins. The separate outer calibration pool still contains only 9 joint blocks. The report includes coverage, width and errors by variable and forecast horizon. These are short synthetic results; there is no coverage guarantee for dependent data. Interval widening leaves P50 unchanged and does not correct the residual sampler's tail distribution.

## Reserve comparison on later synthetic validation

Fixed reserve selected from earlier validation: **0.2 × effective capacity**. Forecast fit and residual blocks both precede tuning. Later evaluation is 72 hours and uses a different partition/model than RESULTS.md; absolute EENS values across these experiments must not be ranked against each other.

| Controller/counterfactual | EENS (kWh) |
|---|---:|
| rule_based | 65.581 |
| deterministic_no_reserve | 66.172 |
| validation_tuned_fixed | 88.458 |
| adaptive_max_fallback | 126.402 |
| adaptive_no_infeasible_target | 91.543 |
| scenario_mpc | 70.322 |
| perfect_foresight_non_deployable | 65.581 |

The maximum-energy infeasibility fallback is detrimental in this validation check. Removing infeasible soft targets reduces EENS, but still does not beat rule based. This counterfactual is a diagnostic result, not a justification to discard infeasible scenarios or a validated controller redesign. Perfect foresight is non-deployable; scenario MPC retains optimistic two-stage recourse.

## What remains incomplete

A permitted, timestamped Indian demand history is still required to align the weather and run Indian sizing, forecasting and reserve experiments. The current workflow uses simplified generation conversion and must not be described as a validated site model. A published, independently tested LFP ageing model and subsequent converged multi-seed SAC study remain incomplete. The original research hypothesis remains unproven.

See [india_workflow.md](india_workflow.md) for exact commands and operational limitations. Evidence files preserve executed summaries in Git; raw weather, checkpoints and full generated traces remain outside version control.
