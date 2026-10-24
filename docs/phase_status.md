# Implementation versus research completion

| Phase | Delivered implementation | Remaining research evidence |
|---|---|---|
| 0 | Mathematical specification, metrics, hypotheses, frozen demo manifests | Empirical feasibility and preregistered final design |
| 1 | Synthetic/CSV pipeline, audits, Parquet, splits, weather conversion, train-only sizing utility | Named public dataset and licence audit |
| 2 | Projected battery simulator, SOH capacity, Gymnasium, randomized physical tests | Historical-site validation |
| 3 | Rule-based, deterministic LP MPC, complete-episode perfect foresight, metrics | Tuning and real-data comparisons |
| 4 | Nine pooled horizon-aware LightGBM quantile models, causal rolling inference, saved model, metrics | Model selection and real-data calibration assessment |
| 5 | Joint blocks, independent ablation, reserve recursion, explicit infeasibility, independent LP oracle | Operational benefit go/no-go on real validation data |
| 6 | Rainflow cycle accounting, illustrative cycle/calendar sensitivity model | Published LFP parameterization, independent empirical validation, trajectory ranking |
| 7 | Vanilla/fixed/adaptive SAC, nested causal chronology, checkpoints, monitor logs, evaluation | Convergence, validation tuning, multi-seed stability |
| 8 | Episode-dual constrained SAC and shared-first-stage two-stage scenario MPC | Constraint tuning, reliability compliance and full comparative study |
| 9 | Seven-controller synthetic benchmark, paired bootstrap, independent-reserve ablation, shortage check, size/risk/seed sweep runner | Full RL ablation matrix, matched-reliability Pareto evaluation, untouched empirical final test |
| 10 | Reproduction commands, dependency lock, generated tables/plot, honest manuscript scaffold | Verified literature, completed research figures/results and conference-ready manuscript |

No phase is represented as scientifically complete solely because its software runs. See `validation.md` for actual executed checks.
