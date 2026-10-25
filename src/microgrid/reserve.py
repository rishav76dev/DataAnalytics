from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from .physics import BatteryConfig


@dataclass(frozen=True)
class Requirement:
    energy_kwh: float
    feasible: bool
    reason: str = ""


def backward_requirement(net_load, config=BatteryConfig()):
    """Minimum absolute starting energy, terminal energy >= physical minimum.

    Curtail surplus as needed; no load shedding allowed in the reserve calculation.
    Infeasibility is explicit rather than clipping the reserve to battery capacity.
    """
    net = np.asarray(net_load, dtype=float)
    if net.ndim != 1 or not len(net) or not np.isfinite(net).all():
        raise ValueError("Need a finite, nonempty net-load trajectory")
    c = config
    needed = c.minimum
    for power in net[::-1]:
        if power > c.power_kw + 1e-9:
            return Requirement(float("inf"), False, "discharge_power")
        if power >= 0:
            needed += power * c.dt_hours / c.discharge_efficiency
        else:
            needed = max(
                c.minimum, needed - min(-power, c.power_kw) * c.dt_hours * c.charge_efficiency
            )
        if needed > c.maximum + 1e-8:
            return Requirement(float("inf"), False, "energy_capacity")
    return Requirement(float(needed), True)


def optimization_oracle(net_load, config=BatteryConfig()):
    """Independent LP minimizing initial energy with hard service of all demand."""
    net = np.asarray(net_load, dtype=float)
    if net.ndim != 1 or not len(net) or not np.isfinite(net).all():
        raise ValueError("Need finite net-load trajectory")
    c, n = config, len(net)
    # Variables E[0:n+1], charge[0:n], discharge[0:n].
    bounds = [(c.minimum, c.maximum)] * (n + 1)
    bounds += [(0, min(max(-p, 0), c.power_kw)) for p in net]
    bounds += [(max(p, 0), max(p, 0)) for p in net]
    if np.any(net > c.power_kw + 1e-9):
        return Requirement(float("inf"), False, "discharge_power")
    a = np.zeros((n, 3 * n + 1))
    for t in range(n):
        a[t, t], a[t, t + 1] = -1, 1
        a[t, n + 1 + t] = -c.charge_efficiency * c.dt_hours
        a[t, 2 * n + 1 + t] = c.dt_hours / c.discharge_efficiency
    objective = np.zeros(3 * n + 1)
    objective[0] = 1
    result = linprog(objective, A_eq=a, b_eq=np.zeros(n), bounds=bounds, method="highs")
    if not result.success:
        return Requirement(float("inf"), False, "energy_capacity")
    return Requirement(float(result.x[0]), True)


class ReserveEstimator:
    def __init__(self, config=BatteryConfig(), alpha=0.05):
        if not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        self.config, self.alpha = config, alpha

    def estimate(self, scenarios):
        s = np.asarray(scenarios, dtype=float)
        if s.ndim != 3 or s.shape[-1] != 3 or min(s.shape[:2]) == 0:
            raise ValueError("Expected (scenario, horizon, [pv, wind, load])")
        if not np.isfinite(s).all() or (s < 0).any():
            raise ValueError("Scenarios must be finite and nonnegative")
        outcomes = [backward_requirement(x[:, 2] - x[:, 0] - x[:, 1], self.config) for x in s]
        requirements = np.array([r.energy_kwh for r in outcomes])
        # Empirical inverse CDF over ALL outcomes, including +inf mass.
        k = int(np.ceil((1 - self.alpha) * len(requirements))) - 1
        target = np.sort(requirements)[k]
        return {
            "target_kwh": float(target) if np.isfinite(target) else None,
            "risk_feasible": bool(np.isfinite(target)),
            "infeasible_fraction": float(np.mean(~np.isfinite(requirements))),
            "power_infeasible": sum(r.reason == "discharge_power" for r in outcomes),
            "energy_infeasible": sum(r.reason == "energy_capacity" for r in outcomes),
            "scenario_count": len(outcomes),
            "alpha": self.alpha,
        }


def end_of_step_targets(scenarios, estimator):
    """Required energy AFTER each action, computed from the remaining suffix.

    Return both uncapped diagnostics and explicitly marked fallback targets.
    The last target is the terminal physical minimum. No infinite value is hidden.
    """
    targets, infeasible = [], []
    for t in range(scenarios.shape[1] - 1):
        result = estimator.estimate(scenarios[:, t + 1 :])
        bad = result["target_kwh"] is None
        targets.append(estimator.config.maximum if bad else result["target_kwh"])
        infeasible.append(bad)
    return np.array([*targets, estimator.config.minimum]), [*infeasible, False]


def vectorized_suffix_reserves(scenarios, config=BatteryConfig(), alpha=0.05):
    """Same recursion, vectorized over scenarios; all suffixes in O(N*H)."""
    s = np.asarray(scenarios, float)
    if s.ndim != 3 or s.shape[-1] != 3 or min(s.shape[:2]) < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid scenario tensor or risk")
    if not np.isfinite(s).all() or (s < 0).any():
        raise ValueError("Scenarios must be finite and nonnegative")
    c = config
    net = s[:, :, 2] - s[:, :, 0] - s[:, :, 1]
    n, h = net.shape
    needs = np.full(n, c.minimum)
    values = np.empty((n, h + 1))
    values[:, h] = needs
    bad = np.zeros(n, bool)
    for t in range(h - 1, -1, -1):
        p = net[:, t]
        needs = np.where(
            p >= 0,
            needs + np.maximum(p, 0) * c.dt_hours / c.discharge_efficiency,
            np.maximum(
                c.minimum,
                needs
                - np.minimum(np.maximum(-p, 0), c.power_kw) * c.dt_hours * c.charge_efficiency,
            ),
        )
        bad |= (p > c.power_kw + 1e-9) | (needs > c.maximum + 1e-8)
        values[:, t] = np.where(bad, np.inf, needs)
    k = int(np.ceil((1 - alpha) * n)) - 1
    targets = np.sort(values, axis=0)[k]
    flags = ~np.isfinite(targets)
    fraction = np.mean(~np.isfinite(values), axis=0)
    # Absolute initial target plus end-of-step suffix targets, including E_min terminal.
    return targets, flags, fraction
