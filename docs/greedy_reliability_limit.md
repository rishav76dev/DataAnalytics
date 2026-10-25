# Why reserve guidance cannot improve the current greedy EENS optimum

This is an exchange argument for the implemented fixed-physics model, not a general statement about real batteries or a literature novelty claim. It explains why increasing SAC training alone cannot establish the proposed reliability advantage.

## Scope

Assume a single islanded bus, inelastic demand, renewable-only charging, no grid or generator, fixed battery capacity and power, constant charge/discharge efficiencies, no self-discharge, equal value per kWh of served load, and a terminal energy requirement of only the physical minimum. Initial energy is identical across policies. Current hourly net demand is known at dispatch, as in the environment. Ageing costs are excluded from the **EENS-only** objective and SOH does not change during this comparison.

## Reduction to a finite energy buffer

Let `z = eta_discharge * (E - E_min)` be stored energy expressed as deliverable AC kWh. Its capacity is `B = eta_discharge * (E_max - E_min)`.

At a surplus hour, the maximum arrival to this buffer is:

```text
a_t = min(renewable_surplus_kw, charge_power_kw)
      * eta_charge * eta_discharge * dt_hours
```

At a deficit hour, service is bounded by stored `z`, the deficit energy, and discharge power. Any demand above the discharge power limit is unavoidable unserved energy for every controller. There is no simultaneous net surplus and net deficit in one time step.

Greedy admits all available surplus up to buffer capacity and serves all currently serviceable deficit. Curtailing an admissible arrival cannot increase later feasible service because storage has no holding penalty in this objective. Deferring a unit of currently feasible service to a later deficit cannot increase total service: move that same unit of service back to the earlier deficit, reducing intermediate inventory and weakly increasing room for intervening arrivals. It respects power limits because the earlier service was feasible by assumption. If an additional arrival is admitted into the freed space, total later service can improve; otherwise the exchange leaves total service unchanged. Repeating these exchanges yields a work-conserving greedy schedule without reducing total served energy.

Consequently, under these assumptions, greedy achieves an EENS optimum. Other schedules can tie it, but forecast uncertainty, a reserve penalty or a learned policy cannot lower EENS below that optimum. A strict terminal-energy requirement changes the feasible set and is outside this claim.

## Checks and interpretation

The project tests greedy against an independently formulated EENS-only LP over 60 randomized parameter/profile cases, varying capacity, power, SOH, timestep, efficiencies and initial SOC. This numerical check supports implementation consistency; it does not replace the argument above or constitute field validation.

On the frozen constructed Indian reference test, the equality between greedy and perfect foresight must be interpreted in this structural context. A forecaster can make an optimization policy worse by unnecessarily retaining energy. An apparent reliability improvement over greedy should first trigger an audit for inconsistent physics, boundary conditions, information or metrics.

## What this does and does not settle

This does **not** prove greedy minimizes calendar damage, cycling damage, curtailment, critical-load outages or a weighted multi-objective cost. Among EENS-optimal schedules there can be different SOC exposure, terminal inventory and damage. A useful reserve-based contribution would therefore need to establish a degradation advantage at matched reliability, or study a separately justified model where reliability decisions are nontrivial.

Capacity fade, temperature-dependent efficiency/power, flexible loads, multiple storage devices, critical-load priorities and uncertain within-hour behavior can invalidate the assumptions. They should be introduced only when supported by the intended application, not added after an unfavorable result merely to force an advantage. The current study does not demonstrate such an extension or an adaptive lifecycle benefit.
