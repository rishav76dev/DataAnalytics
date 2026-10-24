import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .physics import BatteryConfig


def rule_based(current):
    return float(current[2] - current[0] - current[1])


def scenario_mpc(scenarios, energy, config=BatteryConfig(), reserve_kwh=None, reserve_penalty=0.1):
    """Two-stage stochastic LP; shared first charge/discharge, recourse thereafter.

    Recourse has perfect scenario knowledge after the first action; this is an
    optimistic two-stage benchmark, not a full multistage scenario tree.
    Penalties: unserved energy 1000, curtailment .001, throughput .01.
    """
    s = np.asarray(scenarios, dtype=float)
    if s.ndim != 3 or s.shape[2] != 3 or min(s.shape[:2]) == 0:
        raise ValueError("Expected (scenario, horizon, 3)")
    if not np.isfinite(s).all() or (s < 0).any():
        raise ValueError("Scenarios must be finite and nonnegative")
    c = config
    if not c.minimum - 1e-8 <= energy <= c.maximum + 1e-8:
        raise ValueError("Initial energy outside bounds")
    count, h, _ = s.shape
    targets = None
    if reserve_kwh is not None:
        targets = np.broadcast_to(np.asarray(reserve_kwh, dtype=float), (h,))
        if not np.isfinite(targets).all():
            raise ValueError("Reserve must be finite")
    # Per-scenario charge, discharge, shed, curtail, end energy, reserve slack.
    width = 6 * h
    a = lil_matrix((count * 2 * h + 2 * (count - 1), count * width))
    b = np.zeros(a.shape[0])
    objective = np.zeros(count * width)
    bounds = []
    for j, x in enumerate(s):
        base = j * width
        net = x[:, 2] - x[:, 0] - x[:, 1]
        bounds += [(0, min(c.power_kw, max(-v, 0))) for v in net]
        bounds += [(0, min(c.power_kw, max(v, 0))) for v in net]
        bounds += [(0, max(v, 0)) for v in net]
        bounds += [(0, max(-v, 0)) for v in net]
        bounds += [(c.minimum, c.maximum)] * h + [(0, None)] * h
        for t in range(h):
            ch, dis, shed, cur, e, slack = [base + z * h + t for z in range(6)]
            row = j * 2 * h + t
            a[row, dis], a[row, ch], a[row, shed], a[row, cur] = 1, -1, 1, -1
            b[row] = net[t]
            row += h
            a[row, e], a[row, ch], a[row, dis] = (
                1,
                -c.charge_efficiency * c.dt_hours,
                c.dt_hours / c.discharge_efficiency,
            )
            if t:
                a[row, e - 1] = -1
            else:
                b[row] = energy
            objective[[ch, dis]] = 0.01 * c.dt_hours / count
            objective[shed] = 1000 * c.dt_hours / count
            objective[cur] = 0.001 * c.dt_hours / count
            objective[slack] = reserve_penalty / count if reserve_kwh is not None else 0
        if j:
            row = count * 2 * h + 2 * (j - 1)
            a[row, base], a[row, 0] = 1, -1
            a[row + 1, base + h], a[row + 1, h] = 1, -1
    upper = lil_matrix((count * h, count * width)) if reserve_kwh is not None else None
    if upper is not None:
        for j in range(count):
            for t in range(h):
                upper[j * h + t, j * width + 4 * h + t] = -1
                upper[j * h + t, j * width + 5 * h + t] = -1
    result = linprog(
        objective,
        A_eq=a.tocsr(),
        b_eq=b,
        A_ub=upper.tocsr() if upper is not None else None,
        b_ub=np.tile(-targets, count) if upper is not None else None,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"MPC failed: {result.message}")
    return float(result.x[h] - result.x[0])


def deterministic_mpc(forecast, energy, config=BatteryConfig(), **kwargs):
    return scenario_mpc(np.asarray(forecast)[None], energy, config, **kwargs)
