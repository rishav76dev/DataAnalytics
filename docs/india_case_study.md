# India case-study data and experiment plan

Geographic decision: India, as requested. Status: five years of NASA POWER weather have now been downloaded and audited; a scaled national-demand mirror now supports an executed constructed reference experiment. Primary-source verification and a local measured demand series remain outstanding. See [../INDIA_RESULTS.md](../INDIA_RESULTS.md). See [india_workflow.md](india_workflow.md) for implemented commands and executed status. All existing recorded results remain the October 24 synthetic baseline.

## Reference case

Use Jaisalmer, Rajasthan, as the initial geographic reference for a constructed islanded PV–wind–battery microgrid. Proposed coordinates are approximately **26.9157° N, 70.9083° E**. They identify a reference location rather than a measured microgrid installation. Refine the exact weather grid cell and site description before acquisition.

Aim for five complete overlapping years if available, with at least three years for the first study. Choose exact dates based on the intersection of weather and suitable demand records before inspecting controller outcomes. Retain chronological 60/15/10/15 partitions; report which seasons each partition actually covers. A chronological split does not guarantee equal seasonal representation.

## Candidate official sources

These are source candidates, not verified dataset holdings or licence approvals. Check current documentation, coverage, access and terms when acquiring files.

| Requirement | Candidate source | Suitability and unresolved checks |
|---|---|---|
| Hourly irradiance, temperature and wind | [NASA POWER](https://power.larc.nasa.gov/) and its [API documentation](https://power.larc.nasa.gov/docs/services/api/) | Check hourly parameter availability, units, UTC time option, missing-value codes, provenance and acknowledgement requirements for the chosen coordinates and dates. Weather estimates are not measured PV/wind generation. |
| Alternative reanalysis weather | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) | Evaluate ERA5 hourly weather and wind heights; account registration, licence acceptance and API access may be required. Record exact dataset/version and retrieval metadata. |
| Indian power-system demand reference | [Grid Controller of India](https://grid-india.in/) and [Central Electricity Authority](https://cea.nic.in/) | Verify whether downloadable historical hourly or subhourly load exists for the required period and region. Daily reports or aggregate energy totals alone do not supply hourly community demand. |
| Local community or feeder demand | A documented Indian institutional, utility or open research dataset | Prefer measured timestamped kW with clear site metadata, weather overlap, consent where applicable and explicit reuse terms. No suitable file has yet been selected. |

Public access is not automatically permission to redistribute. Archive source URLs, retrieval time, licence/terms text and raw-file SHA-256 hashes. Keep raw files unchanged. Do not commit third-party files until their redistribution rights are established.

## Demand is the critical selection gate

Prefer geographically relevant measured community/feeder demand. State-level demand cannot be presented as observed microgrid load. If only aggregate Indian demand is available and its terms permit reuse, a scaled profile may be used as an explicitly **constructed Indian reference-load scenario**, with the scaling fit on training only and sensitivity analysis for peak-to-average shape. Aggregation removes local variability, so this option cannot establish community-level reliability.

If no defensible hourly demand series is available, complete weather auditing independently and report the demand-data blocker. Synthetic demand may support further software checks, but must remain labelled synthetic; do not relabel it historical Indian demand.

## Time and unit alignment

1. Preserve the original timestamps and source timezone. India uses `Asia/Kolkata` (UTC+05:30), without daylight-saving transitions in the modern study period.
2. Request weather in UTC when supported. NASA POWER's local solar time is not IST; do not convert it by simply attaching `Asia/Kolkata`.
3. Convert explicitly zoned source timestamps to UTC for storage and joining. Retain local IST hour/calendar features when justified by Indian demand behavior; current forecast calendar features use stored timestamps and need a documented implementation change before claiming local-time features.
4. Distinguish interval-start, interval-end and instantaneous measurements. Decide a common hourly aggregation convention before joining. IST clock-hour bins are offset by 30 minutes from UTC clock-hour bins; changing timezone alone does not align those integration windows. Use documented overlap-weighted energy aggregation when source resolution permits, otherwise disclose the approximation or reject incompatible records.
5. Convert interval energy to mean power using the actual interval duration. Do not interpret kWh or MW as kW without conversion. Check wind-height and irradiance-unit metadata before applying physical models.
6. Report duplicate timestamps, gaps, source quality flags and missing-value sentinels. Predeclare gap repair and reject unresolved prolonged gaps. The current ingestion API accepts continuous hourly `timestamp,pv_kw,wind_kw,load_kw`; it does not perform these source-specific conversions automatically.

## Weather-to-generation assumptions

Use a documented PV model and a specified wind turbine/power curve. The current simplified conversion utilities are engineering starting points, not site-validated plant models. Record PV tilt/orientation, inverter/system loss assumptions, wind hub height, any vertical extrapolation and cut-in/rated/cut-out speeds. Wind measured or modelled at 10 m cannot be treated as hub-height wind without an explicit assumption and sensitivity analysis.

Derive reference renewable capacities from training only. Freeze those choices before validation and test comparisons. Keep the default 100 kWh / 50 kW battery as a baseline candidate; establish adequacy rather than assuming it is suitable for this Indian case.

## Ordered experiments and acceptance gates

| Stage | Required deliverable before moving on |
|---|---|
| Data | Named weather and demand datasets, permitted usage, raw hashes, UTC-aligned hourly Parquet, audit report and fixed split boundaries |
| Feasibility | Battery kWh/kW and renewable-capacity sweeps; rule-based and complete-episode perfect-foresight EENS/LPSP; shortage causes; initial/terminal SOC sensitivity |
| Uncertainty | Held-out interval coverage by source and horizon, daylight-only solar diagnostics, calibrated versus raw forecasts, historical residual block counts and shortage-tail checks |
| Reserve | Tuned fixed-reserve comparison on validation, explicit power/capacity infeasibility, no-reserve and fallback counterfactuals, paired uncertainty and matched-reliability throughput |
| Lifecycle/RL | Verified published LFP model first; then validation-selected SAC budgets, multiple seeds and frozen longer final-test evaluation |

For Rajasthan, inspect pre-monsoon, monsoon, post-monsoon and winter periods separately, using documented month definitions. Examine consecutive low-generation episodes and high-demand periods in the actual records; do not presume seasonal adequacy from annual mean generation.

The provisional target remains 1% energy LPSP. Perfect-foresight feasibility is necessary evidence for a candidate design, not a deployable performance guarantee. A reserve that is infeasible must remain reported; do not change risk tolerance simply to suppress the diagnostic.

## Relationship to current code and outputs

The existing CSV/Parquet ingestion and controllers can be reused after source-specific preprocessing passes the audit. The new [executable workflow](india_workflow.md) adds an India-weather downloader and validation commands. Forecast calendar features still use stored timestamps; no demand proxy has been validated and no Indian controller results have been produced. See [next_experiment.md](next_experiment.md) for the full experimental gates and [../RESULTS.md](../RESULTS.md) for the preserved synthetic baseline.

The available synthetic review archive remains local at `outputs/audit/baseline_review.zip`. It contains traces and checksums for baseline diagnosis, not Indian historical observations.
