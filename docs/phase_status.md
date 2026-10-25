# Implementation versus research completion

Updated after the constructed Indian reference experiment: [results](../INDIA_RESULTS.md), [reproduction](india_research_execution.md). These software and experimental deliverables do not establish adaptive superiority or a submission-ready field study.

| Phase | Delivered implementation | Remaining research evidence |
|---|---|---|
| 0 | Mathematical specification, metrics, hypotheses, frozen demo manifests | Empirical feasibility and preregistered final design |
| 1 | NASA weather download, pinned national-demand reference adapter, audits, Parquet, splits and train-only sizing | Primary demand-record/terms verification and measured local demand |
| 2 | Projected battery simulator, SOH capacity, Gymnasium, randomized physical tests | Historical-site validation |
| 3 | Rule-based, deterministic LP MPC, complete-episode perfect foresight, metrics | Tuning and real-data comparisons |
| 4 | Nine pooled horizon-aware LightGBM quantile models, causal rolling inference, saved model, metrics | Model selection and real-data calibration assessment |
| 5 | Joint blocks, independent ablation, reserve recursion, explicit infeasibility, independent LP oracle | Operational benefit go/no-go on real validation data |
| 6 | Published BLAST-Lite LFP parameter model, pinned-reference regression, rainflow, offline ranking and capacity-feedback sensitivity | Pack-specific empirical validation and applicability outside tested cell conditions |
| 7 | Four SAC variants, nested causal chronology, three seeds, 10k/30k budgets, validation-selected checkpoints and test evaluation | Convergence and generalization remain unestablished; policy suffix lacks full seasonal coverage |
| 8 | Three-seed constrained SAC and shared-first-stage scenario MPC evaluated | Reliability target remains unmet; no guarantee from learned penalty |
| 9 | Constructed reference full-test comparison, three-seed RL, sizing, bootstrap block sensitivity, independent-reserve and fallback counterfactuals, modelled-damage plot | Complete ablation matrix, primary-source verification and field validation; adaptive matched-reliability benefit not demonstrated |
| 10 | Reproduction commands, dependency lock, committed evidence, generated tables/plot and compiled negative-result report | Comprehensive novelty review, primary provenance and submission-ready scientific claims |

No phase is represented as scientifically complete solely because its software runs. See `validation.md` for actual executed checks.
