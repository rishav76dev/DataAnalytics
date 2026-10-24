# Executed validation

Executed on 2026-10-24 in the cloud checkout. These are synthetic engineering results, not empirical research findings.

- Frozen dependency installation including CPU SAC dependencies: passed.
- Pytest: **18 passed**, including SAC checkpoint save/load. Gymnasium emitted advisory warnings for unbounded observation-space declarations and an unregistered render spec; no test failed.
- Ruff check of src, tests and scripts: passed.
- Documented adaptive train/evaluate CLI and the 24-hour benchmark sweep entry point: passed.
- End-to-end default demo: 2,160 generated hours; 72 evaluation hours; 100 synchronized scenarios per origin; seven controller traces and summary outputs generated.
- All four SAC variants: 1,500 training steps each, seed 42, separate held-out evaluation; checkpoints and metrics generated. This is a smoke run, not convergence evidence.

| Controller | EENS (kWh) | Energy LPSP |
|---|---:|---:|
| rule_based | 97.342 | 0.1125 |
| deterministic_mpc | 97.587 | 0.1128 |
| fixed_reserve_mpc | 114.477 | 0.1323 |
| adaptive_reserve_mpc | 119.120 | 0.1377 |
| independent_reserve_mpc | 113.823 | 0.1315 |
| scenario_mpc | 97.342 | 0.1125 |
| perfect_foresight_non_deployable | 97.342 | 0.1125 |

Nominal 80% quantile coverage on validation was **64.3%**; the model is undercovered and needs calibration before a reliability claim.

The 95th-percentile reserve was infeasible at **63 of 72** evaluated origins. Those outcomes remain explicit in reserves.json. The demo does not establish the provisional 1% LPSP target or adaptive superiority.

Generated plots, models, metrics and traces are retained under ignored outputs/. The source, configuration and uv.lock are available for review; no upstream publication or real-data experiment is implied.
