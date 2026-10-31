# Recorded results: degradation-aware adaptive-reserve MPC

The frozen computational experiment is reproducible, but **robust ageing superiority at matched reliability has not been demonstrated**. This document summarizes the current study; [RESULTS.md](RESULTS.md) preserves the older synthetic experiment.

Use [testscript.md](testscript.md) to check the numbers against the committed hourly traces, rather than relying only on this summary. The numerical experiment and evidence were released in commit `4750666732902d800075c1a24123efb872c2ff09`.

## Experimental conditions

| Item | Recorded setting |
|---|---|
| Case | Jaisalmer NASA POWER weather plus scaled Indian national demand; constructed reference, not a measured campus/village microgrid |
| Forecaster fitting | 2022 |
| Residual scenario library | 2023 |
| Initial online calibration history | January–June 2024 |
| Controller and hardware selection | July–December 2024 |
| Frozen holdout | January 1–June 24, 2025, in Asia/Kolkata local time |
| Holdout duration | 4,200 hours: 4,176 operational hours plus 24 hours of common terminal normalization |
| Battery | 400 kWh, 50 kW, SOC 10–90%, initial SOC 50%, charge/discharge efficiencies 95% |
| Renewable nameplates | PV 95.653 kW; wind 126.764 kW; scaling fitted on training data only |
| Operational demand | 58,652.039 kWh |
| Primary ageing evaluation | Published BLAST-Lite LFP-Gr 250Ah model, assumed battery temperature 25°C, 24-hour updates, exact effective-age integration |
| Statistical comparison | Paired circular block bootstrap, 1,000 replicates, 24-hour and 168-hour blocks |

The final 24 hours use the same recharge-only policy for every controller: renewable surplus charges the battery and battery discharge is disabled. Every primary controller finishes at **360 kWh**. This normalization adds **9.1599 kWh of unserved energy for every controller**. Operational EENS excludes that day; ageing includes it. Thus zero operational EENS does **not** mean zero EENS over all 4,200 hours.

## Controller results

Capacity loss below is a **model prediction**, not a measured ageing result or a percentage extension in battery life.

| Controller | Operational EENS (kWh) | Operational energy LPSP (%) | Predicted capacity loss (%) | Matches reference reliability and inventory? |
|---|---:|---:|---:|:---:|
| Greedy reference | 0.000 | 0.0000 | 1.2376 | Yes |
| Tuned fixed-reserve ageing MPC | 0.000 | 0.0000 | 1.2360 | Yes |
| **Joint adaptive-reserve ageing MPC** | **0.000** | **0.0000** | **1.2140** | **Yes** |
| Independently calibrated adaptive reserve | 12.308 | 0.0210 | 1.1934 | No |
| Scenario MPC with the same envelope | 0.000 | 0.0000 | 1.2148 | Yes |
| Tuned fixed reserve with weekly-persistence forecasts | 0.000 | 0.0000 | 1.2370 | Yes |
| Adaptive reserve without calendar cost | 0.000 | 0.0000 | 1.2280 | Yes |
| Adaptive reserve with uncorrected envelope | 739.377 | 1.2606 | 1.0017 | No |
| Point-forecast ageing MPC | 16,282.096 | 27.7605 | 0.3640 | No |
| Weekly-persistence point ageing MPC | 4,665.494 | 7.9545 | 0.6957 | No |
| Scenario ageing MPC | 4,017.585 | 6.8499 | 0.7621 | No |

Source: [controller_results.csv](docs/evidence/age_mpc_v2/controller_results.csv) and [test_metrics.json](docs/evidence/age_mpc_v2/test_metrics.json). Full traces and their checksums are [committed here](docs/evidence/age_mpc_v2/README.md).

## Interpretation of the hypothesis test

Adaptive MPC predicts **1.7825% less capacity loss relative to tuned fixed reserve**, calculated as `100 × (1 − 1.2140114 / 1.2360437)`. This is about **0.0220 percentage points of capacity loss**, not a 1.78 percentage-point change. Both controllers have zero operational EENS and the same terminal inventory. The daily paired damage interval excludes zero, but the weekly interval includes zero. The predeclared confidence requirement therefore fails.

The independently calibrated reserve was the lowest-damage eligible baseline on validation data, so it remains the **primary comparator**. It predicts less damage on the test but incurs 12.308 kWh operational EENS. That exceeds the permitted matching margin of **5.8652 kWh** (0.01 percentage points of operational energy LPSP). This comparison does not establish matched-reliability superiority. The primary comparator was not replaced after seeing its test performance.

The scenario-plus-envelope baseline is also competitive: the proposed method's relative damage reduction is only **0.0670%**, and both block-bootstrap damage intervals include zero. Lower-damage controllers that shed substantially more load cannot support a matched-reliability ageing claim.

See [paired comparisons](docs/evidence/age_mpc_v2/matched_reliability.json) and [claim gates](docs/evidence/age_mpc_v2/claim_gates.json). These intervals describe variability within this constructed episode; they do not include uncertainty in battery parameters or constitute independent site replications.

## Calibration, data and battery checks

- Simultaneous coverage of all contiguous future DC-deficit intervals improved from **19.34% raw to 91.26% corrected**. The aggregate exceeds the study's 90% screening gate but misses the nominal 95% target. February coverage is only **70.39%**, so temporal calibration remains unresolved.
- Future-lead marginal interval diagnostics achieve about **75.11% coverage** against a nominal 80%. Learned pooled future MAE is **5.299 kW**, versus **4.591 kW** for weekly persistence. Forecast superiority is not established.
- The primary 2024–2025 demand workbook has **12,984 hourly records**, exactly matching the pinned mirror on the overlap. Publisher checksum, MW units and CC BY 4.0 metadata were verified. Earlier demand remains a pinned mirror, and the naive source timestamps require an explicit IST interpretation.
- All **175 primary adaptive ageing windows** fall below the published model's fitted DOD range. Numerical implementation checks pass, but **empirical battery/pack applicability is not validated**.
- Six frozen-policy capacity-fade sensitivities ran at 25°C and 35°C. Joint adaptive MPC and greedy retain zero operational EENS; independent reserve incurs **12.679 / 12.922 kWh**, respectively. These sensitivities do not repair the failed primary superiority comparison.

## Verification status and research decision

On October 31, 2026, the documented verification commands were rerun: **43 automated tests passed**, the release integrity verifier passed, and the extended trace checks reproduced reliability, energy accounting, ageing predictions, calibration rates and the key paired comparisons.

Passing these checks means the recorded calculations agree with the released evidence and implementation. It does not establish that the constructed load is representative of a local site, that the ageing model predicts a real pack accurately, or that the method is novel enough for an IEEE paper.

Further SAC research training remains paused. The next confirmatory study needs an independently sourced Indian site load, credible generation conversion, a shallow-cycle-applicable LFP model or independent measurements, improved calibration under drift, and untouched seasons/sites. See [full report](MPC_RESEARCH.md), [protocol](docs/degradation_mpc_protocol.md), [validation card](docs/age_mpc_validation.md) and [reproduction guide](docs/age_mpc_reproduction.md).
