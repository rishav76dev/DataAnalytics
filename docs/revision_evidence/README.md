# Limited-revision evidence

Run the script at scripts/age_mpc_revision_evidence.py from the repository root with pandas, pyarrow, numpy, rainflow and matplotlib installed. It reads only the immutable release under docs/evidence/age_mpc_v2. It asserts the 4,176-hour operational/24-hour normalization split, EENS totals and 360 kWh terminal inventory before writing:

- matched_comparison.csv: the preselected independent adaptive comparator and the reliability-matched greedy, fixed, proposed and scenario-envelope group. The paired upper bounds here compare each method **with greedy**. The preselected comparator remains the primary hypothesis comparison even though its 12.308 kWh operational EENS exceeds the 5.8652 kWh matching allowance.
- adaptive_direct_comparisons.csv: adaptive-minus-fixed, adaptive-minus-scenario-envelope and adaptive-minus-preselected-independent paired intervals, with the original 24-hour and 168-hour blocks and 1,000 replications.
- battery_stress.csv: operational throughput and SOC exposure; all-hours rainflow equivalent full cycles and cycle-depth counts. A half cycle counts as 0.5. Reserve request and attainment fields are given only where the trace stores a dynamic target; the fixed-reserve trace uses default request fields and must not be interpreted as a measured fixed target.
- monthly_coverage.pdf: raw and corrected simultaneous deficit-envelope coverage. February corrected coverage is 70.39%, despite 91.26% overall. June is partial.

The original controller results and claim gates remain in docs/evidence/age_mpc_v2. No frozen result is overwritten. These files are derived reporting artifacts.

## Battery-model decision

BLAST-Lite's fitted 80–100% depth-of-discharge range excludes all 175 primary adaptive daily windows; its capacity-loss values remain exploratory. We screened the independent [Wang et al. LFP cycle-life model](https://doi.org/10.1016/j.jpowsour.2010.11.134), whose [published paper](https://eda.ee.ucla.edu/member_only/BMS/references/SOH/cycle%20life%20lifepo4.pdf) describes tests down to 10% depth but no discharge rate below C/2. The studied 50 kW/400 kWh battery has a rated maximum of C/8. A numerical replay under Wang's parameters would still extrapolate across charge/discharge rate, and cycles below 10% depth would remain outside its experimental matrix. We therefore did not create a misleading second capacity-loss series. Throughput, equivalent full cycles, SOC exposure and cycle-depth distribution are the primary battery-stress results. They do not measure pack lifetime.

## Renewable-shortage sensitivity: input boundary and fixed rule

The committed release has dispatch traces but lacks the hourly input parquet, frozen model pickle, development predictions, and test forecast/scenario arrays required by scripts/age_mpc_study.py. Applying a renewable multiplier to **recorded actions** would not rerun the controllers or keep forecast/scenario inputs consistent. Consequently no shortage-controller outcome is reported or implied.

Once those original inputs are restored, select the derating using **July–December 2024 only**: test common PV/wind availability factors 0.95, 0.90, 0.85, ..., 0.50 and choose the largest factor for which validation greedy operational LPSP first reaches at least 0.1% but stays at or below 1%. If none qualifies, report that the predeclared rule found no usable single stress case; do not choose a factor using test outcomes. Freeze the factor, code and original controller parameters before evaluating the 2025 supplementary case. Multiply observed PV and wind, their point forecasts and every sampled scenario by the same factor, then recompute the calibrated deficit-envelope inputs under that stressed validation history. Run greedy, fixed-reserve, adaptive-reserve and matched-envelope scenario MPC with common physical constraints and the same terminal normalization. Report operational/all-hours EENS, LPSP, loss-of-load hours, reserve infeasibility, reserve attainment and SOC. This sensitivity is secondary to the unchanged holdout and is not independent-site validation.

## Runtime scope

Existing p95 controller timings are 7.64 ms for adaptive MPC and 250.74 ms for scenario-envelope MPC. The recorded batch forecast and shared preparation measurements do not establish comparable online pipeline latency. The manuscript treats them as descriptive implementation timings and makes no computational-superiority claim.
