# Indian reference-case research results

The 2022–2024 results below are preserved. The newer [degradation-aware MPC study](MPC_RESEARCH.md) verifies the primary 2024–2025 demand workbook and licence, tests a fresh 2025 holdout, and releases full traces. Earlier primary-source access limitations described here were subsequently resolved for that workbook; site-demand representativeness and pack validation remain unresolved. Further SAC research training is paused.

**Decision: the proposed adaptive reliability–degradation advantage is not demonstrated.** This is a completed computational experiment with negative findings, not a validated Indian field trial or a submission-ready superiority claim.

## Data and source confidence

- 2022–2024 constructed input: **26,304 hourly rows**.
- NASA POWER Jaisalmer weather, aligned to a scaled national Indian demand profile.
- National demand is a pinned secondary mirror attributed to Mendeley DOI `10.17632/y58jknpgs8.2`; its documentation reports MW, IST and CC BY 4.0. The primary Mendeley record remains proxy-blocked, so these claims are explicitly secondary-source evidence.
- Training-only demand normalization: 12 kW mean. Six implausible readings were causally filled within a three-hour maximum; exact hashes and transformations are retained.
- This does not measure an operating local community microgrid. Wind conversion at 10m remains a simplified assumption.

## Frozen experiment

- Chronological 60/15/10/15 split; complete final partition: **3,947 hours** (about 164 days).
- 72 validation configuration/boundary cases: capacity, power, renewable scaling, initial SOC and terminal-energy sensitivities. The expanded search follows failure of the initial designs.
- Selected design: **200 kWh / 50 kW**, renewable capacities **3×** the training-derived reference. This is lexicographic selection, not an economic optimum.
- Fixed reserve selected on validation: **40% of effective capacity**.
- Forecast/scenario setup: 24-hour horizon, 100 sampled scenarios, **109 historical joint residual blocks**, alpha 0.05.
- Data, model and source-code hashes written before conventional final dispatch. The original October 24 synthetic results remain unchanged in [RESULTS.md](RESULTS.md).

## Conventional final-test outcomes

| Controller | EENS (kWh) | Energy LPSP (%) | Published-model capacity loss (%) |
|---|---:|---:|---:|
| rule_based | 527.693 | 1.0554 | 1.30061 |
| deterministic_mpc | 2264.201 | 4.5283 | 1.08179 |
| tuned_fixed_mpc | 1397.912 | 2.7957 | 1.19291 |
| adaptive_max_fallback | 1692.217 | 3.3843 | 1.25122 |
| adaptive_no_infeasible_target | 1735.336 | 3.4706 | 1.23458 |
| independent_reserve_mpc | 2638.940 | 5.2777 | 1.13518 |
| scenario_mpc | 1337.343 | 2.6746 | 1.09514 |
| perfect_foresight_non_deployable | 527.693 | 1.0554 | 0.98826 |

Rule based and the non-deployable perfect-foresight EENS optimum are equal. Both narrowly miss the provisional 1% LPSP target on this held-out episode; no post-test resizing is used to conceal the miss. All forecast-based methods have higher EENS. The oracle does not optimize ageing: its lower modelled damage is an incidental EENS-optimal schedule, not a deployable lifecycle result.

See [the fixed-physics greedy argument](docs/greedy_reliability_limit.md). Under the current EENS-only assumptions, reserve guidance cannot improve on the greedy optimum. It can still affect damage and other objectives, which require matched-reliability evidence.

## Uncertainty and scenario findings

- Nominal 80% marginal interval coverage: raw **74.66%**, corrected **84.11%**.
- Scenario net-energy upper-95% bound coverage: only **75.61%**.
- Infeasible initial-reserve origins: **34.08%**.
- The 24-hour block bootstrap for adaptive-minus-fixed mean hourly unserved power gives [0.00879, 0.13231] kW; 168-hour blocks give [-0.02759, 0.15614] kW. The dependence sensitivity prevents a robust significance claim.
- Marginal interval widening leaves P50/scenario dispatch unchanged. It does not solve scenario-tail calibration. Origin-unconditional daily residuals and the absence of a PV astronomical night mask remain limitations.

## SAC: three seeds and validation-selected budgets

All four variants use seeds 42, 43 and 44; checkpoints are compared at 10,000 and 30,000 training steps. Selection uses validation EENS, then modelled damage, before any RL final-test evaluation. In total this is **360,000 training steps** and **24 validation checkpoint evaluations**. Longer training is not consistently better. This comparison does not establish convergence or seed stability.

| Variant | Seeds | Mean EENS (kWh) | Sample SD (kWh) | Min–max EENS (kWh) |
|---|---:|---:|---:|---:|
| vanilla | 3 | 629.578 | 109.184 | 537.033–749.996 |
| fixed | 3 | 1076.970 | 818.772 | 532.367–2018.556 |
| adaptive | 3 | 604.395 | 106.427 | 539.869–727.234 |
| constrained | 3 | 903.396 | 623.614 | 539.955–1623.473 |

Training uses a nested causal forecast/residual warmup and a common policy-training suffix, which covers only part of the outer training period rather than all seasons. The 12-dimensional observation includes ageing states. All variants share a published-model-derived hourly damage proxy; offline comparison uses daily model updates. The constrained variant adds an episode-wise dual penalty; its reliability target is not guaranteed. Physical battery limits remain hard. Full candidate results, chosen budgets and checkpoint hashes are versioned in the evidence folder.

Across the 12 selected RL trajectories, the hourly published-model reward proxy versus daily offline model damage has Spearman rank correlation **0.818**. This is a descriptive implementation check, not empirical validation or an independent 12-sample battery study.

## Published LFP model and capacity feedback

Implemented the BLAST-Lite 250 Ah LFP-Gr model at pinned commit `b093495b47dc40dd96dba865d91f553619501e94`, whose code cites experimental DOI `10.1016/j.est.2023.109042`. Regression tests agree with the upstream coefficient/state implementation over repeated profiles. BSD notices and reference provenance are retained. This is numerical source verification, not independent laboratory or pack validation.

A separate daily capacity-feedback sensitivity increases rule-based EENS from 527.693 kWh to **599.586 kWh at assumed 25°C**, and **642.530 kWh at assumed 35°C**. Modelled final SOH is 0.98699 and 0.98085. Removed stored energy is explicitly accounted for. This post-hoc sensitivity does not change the primary fixed-capacity or RL comparisons.

Throughput alone ranks published-model damage poorly across the conventional controllers. Raw damage reductions at worse reliability do not support a lifetime-extension claim. Cell-to-pack transfer, daily DOD extrapolation and controlled battery temperature remain limitations.

## Deliverables and reproduction

- [Executed source and workflow](docs/india_research_execution.md)
- [Versioned metrics, source hashes, selections and plots](docs/evidence/india_reference_2026-10-25/)
- [Experimental manuscript source](paper/manuscript.tex)
- [Compiled experimental report](docs/evidence/india_reference_2026-10-25/experimental_report.pdf)
- [Reliability versus modelled damage](docs/evidence/india_reference_2026-10-25/reliability_degradation.png)

The source, summaries and manuscript are retained in Git. Raw weather/demand, full traces, models and checkpoints remain in ignored data/ and outputs/. A local review archive is generated separately with checksums. No publication or external submission is implied.

## Research conclusion

The go/no-go gate is **NO-GO for claiming adaptive superiority in this model and experiment**. A defensible next contribution would have to demonstrate degradation reduction at matched reliability or justify different application physics/objectives; it cannot assume an EENS improvement over an already optimal greedy policy. Primary dataset verification, a proper seasonal uncertainty model, cell-to-pack validation and a complete novelty review remain required before an IEEE submission.
