# Architecture and mathematical contract

```mermaid
flowchart LR
    D[Hourly PV / wind / load] --> A[Audit and chronological splits]
    A --> F[Causal LightGBM quantiles]
    F --> S[Joint calibration residual blocks]
    S --> R[Backward reserve recursion]
    F --> C[MPC / SAC / rule-based controllers]
    R --> C
    C --> P[Physical action projection]
    P --> B[Battery and AC balance]
    B --> E[Reliability / illustrative ageing / runtime]
```

Dispatch duration is Δt = 1 h; horizon H = 24. Nominal battery is 100 kWh / 50 kW; SOC bounds [0.1, 0.9], initial SOC 0.5, charge and discharge efficiency both 0.95. SOH is a fixed input per experiment; effective capacity is nominal capacity × SOH. Offline illustrative ageing is not automatically fed back into SOH. No grid, diesel, flexible demand, simultaneous charge/discharge, or battery-to-curtailment discharge is permitted.

**Sign:** AC bus battery power p > 0 discharges, p < 0 charges. Positive net demand d = load − PV − wind. Project p onto the intersection of power, energy, and net-demand/surplus bounds. The bus identity is PV + wind + p + unserved = load + curtailed. Battery update is E′ = E − pΔt/η_d for discharge and E′ = E − pη_cΔt for charge. Curtailment and unserved energy are reported separately; losses are represented only in battery energy conversion.

**Reserve:** Let R_H = E_min. Backwards for d_h ≥ 0, R_h = R_(h+1) + d_hΔt/η_d, provided d_h ≤ P_max. For d_h < 0, R_h = max(E_min, R_(h+1) − min(−d_h, P_max)η_cΔt). If any R_h exceeds E_max the trajectory is energy-infeasible. Power-infeasible and energy-infeasible outcomes receive infinite required energy. The target is order statistic ceil((1−α)N) of ALL requirements. A null serialized target means the chance requirement is infeasible; no feasible-only conditioning is performed. This is empirical scenario feasibility, not a distribution-free reliability guarantee.

Targets for an action's end energy use the remaining scenario suffix (hours h+1 through H−1), not the reserve required before that action. MPC accepts a time-indexed target vector. When a suffix reserve is infeasible, its flag remains in output and the soft operational fallback is E_max; no physical feasibility is claimed. The final terminal target is E_min.

**Optimization:** LP variables are charge, discharge, unserved power, curtailment, energy, and soft reserve slack. All physical energy bounds remain hard. Costs are 1000 per kWh unserved, 0.01 per kWh AC charge/discharge, 0.001 per kWh curtailed and 0.1 per kWh reserve slack per stage. The perfect-foresight reference optimizes the complete evaluated episode, uses future actual values, and is labelled non-deployable. Scenario MPC shares its first charge/discharge across every scenario. Later recourse has scenario knowledge, so this is an optimistic two-stage benchmark, not a multistage non-anticipative tree.

**Learning:** Action is a scalar in [−1,1] times P_max, followed by the same physical filter. Observations contain SOC, current measured PV/wind/load divided by a fixed 50 kW reference, cyclic hour, reserve/capacity, explicit reserve-infeasible indicator and fixed SOH. Training reward is −(1000×unserved kWh + 0.01×DC throughput kWh + 0.001×curtailed kWh + 0.1×reserve-deficit kWh)/1000. Vanilla omits reserve cost/observation signal; fixed uses 40% effective capacity; adaptive uses causal scenario reserve. Constrained SAC adds λ×unserved power/50 and updates λ ← max(0, λ + 10×(episode LPSP−0.01)). All hard physical safety remains independent of λ.

**Interpretation:** With no price arbitrage, flexible demand or external source, conserving energy is already a strong heuristic. Extra reserve cannot create renewable energy or fix a power-infeasible load. Equal or worse adaptive performance is a valid outcome.
