"""Reliability-first MPC, calibrated reserve trajectories and an LFP wear proxy."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix, vstack

from .ageing import graphite_potential
from .physics import BatteryConfig


def wear_coefficients(
    config=BatteryConfig(), temperature_c=25.0, reference_days=100.0, reference_efc=100.0
):
    """Model-derived positive linear upper envelope for SOC calendar exposure.

    Reference-age marginal cyclic damage is a DC-throughput proxy. Its ranking is
    checked offline; it is not asserted equal to full daily/rainflow degradation.
    """
    if not 10 <= temperature_c <= 45:
        raise ValueError("Unsupported model temperature")
    soc = np.linspace(config.soc_min, config.soc_max, 161)
    kelvin = temperature_c + 273.15
    kcal = 8.37e4 * np.exp(-5210 / kelvin) * np.exp(-3560 * graphite_potential(soc) / kelvin)
    increment = kcal * ((reference_days + 1 / 24) ** 0.526 - reference_days**0.526)
    slope = max(float((increment[-1] - increment[0]) / (soc[-1] - soc[0])), 0.0)
    intercept = float(np.max(increment - slope * soc))
    kcyc = (4.38e-8 + 1.55e-8 * 0.8 + 1.68e-7 * 0.25) * (
        np.exp(2190 / kelvin) + np.exp(-1.55e5 / kelvin)
    )
    return {
        "calendar_per_kwh_hour": slope / config.capacity,
        "calendar_intercept_per_hour": intercept,
        "cyclic_per_dc_kwh": kcyc * 0.828 * reference_efc ** (-0.172) / (2 * config.capacity),
        "upper_envelope_gap_max": float(np.max(intercept + slope * soc - increment)),
        "reference_days": reference_days,
        "reference_efc": reference_efc,
        "temperature_c": temperature_c,
    }


def age_mpc(
    scenarios,
    energy,
    config=BatteryConfig(),
    reserve=None,
    eens_slack_kwh=0.0,
    calendar=True,
    force_full_lp=False,
):
    """Lexicographic service -> reachable reserve -> ageing proxy, common first action.

    All controllers serve feasible current demand before optimizing charging.
    Later scenario recourse is optimistic two-stage, explicitly not multistage.
    """
    s = np.asarray(scenarios, float)
    if s.ndim == 2:
        s = s[None]
    if s.ndim != 3 or s.shape[-1] != 3 or not np.isfinite(s).all() or (s < 0).any():
        raise ValueError("Invalid forecasts")
    if not np.allclose(s[:, 0], s[0, 0], atol=1e-9):
        raise ValueError("First-stage information must be identical")
    c = config
    n, h, _ = s.shape
    if not c.minimum - 1e-8 <= energy <= c.maximum + 1e-8 or eens_slack_kwh < 0:
        raise ValueError("Invalid energy/reliability budget")
    net0 = s[0, 0, 2] - s[0, 0, 0] - s[0, 0, 1]
    # No avoidable present load shedding. This also avoids needless LP solves.
    if net0 >= 0:
        return min(
            net0, c.power_kw, max(0.0, energy - c.minimum) * c.discharge_efficiency / c.dt_hours
        )
    if energy >= c.maximum - 1e-8:
        return 0.0
    targets = np.asarray(reserve, float) if reserve is not None else None
    if targets is not None and (targets.shape != (h,) or not np.isfinite(targets).all()):
        raise ValueError("Invalid reserve path")
    if targets is not None and not force_full_lp:
        available = min(
            c.power_kw, -net0, (c.maximum - energy) / (c.charge_efficiency * c.dt_hours)
        )
        reachable = energy + available * c.charge_efficiency * c.dt_hours
        if targets[0] >= reachable - 1e-9:
            # More surplus charging cannot worsen stage-one service or future
            # lower-reserve slack, and strictly reduces current reserve slack.
            return -float(max(available, 0.0))
    width = 5 * h
    slack_count = h if targets is not None else 0
    size = n * width + slack_count
    a = lil_matrix((2 * n * h + 2 * (n - 1), size))
    b = np.zeros(a.shape[0])
    service = np.zeros(size)
    wear = np.zeros(size)
    reserve_obj = np.zeros(size)
    coeff = wear_coefficients(c)
    bounds = []
    for z, x in enumerate(s):
        base = z * width
        net = x[:, 2] - x[:, 0] - x[:, 1]
        bounds.extend([(0, min(c.power_kw, max(-v, 0))) for v in net])
        bounds.extend([(0, min(c.power_kw, max(v, 0))) for v in net])
        bounds.extend(
            [(0, max(v, 0)) for v in net]
            + [(0, max(-v, 0)) for v in net]
            + [(c.minimum, c.maximum)] * h
        )
        for t in range(h):
            ch, dis, shed, cur, e = [base + k * h + t for k in range(5)]
            row = z * 2 * h + t
            a[row, ch], a[row, dis], a[row, shed], a[row, cur] = -1, 1, 1, -1
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
            service[shed] = c.dt_hours / n
            wear[ch] = coeff["cyclic_per_dc_kwh"] * c.charge_efficiency * c.dt_hours / n * 1e6
            wear[dis] = coeff["cyclic_per_dc_kwh"] / c.discharge_efficiency * c.dt_hours / n * 1e6
            if calendar:
                wear[e] = coeff["calendar_per_kwh_hour"] * c.dt_hours / n * 1e6
        if z:
            row = 2 * n * h + 2 * (z - 1)
            a[row, base], a[row, 0] = 1, -1
            a[row + 1, base + h], a[row + 1, h] = 1, -1
    bounds.extend([(0, None)] * slack_count)
    a = a.tocsr()
    first = linprog(
        service, A_eq=a, b_eq=b, bounds=bounds, method="highs-ds", options={"time_limit": 10.0}
    )
    if not first.success:
        raise RuntimeError(first.message)
    upper = csr_matrix(service[None])
    rhs = np.array([first.fun + eens_slack_kwh + 1e-6])
    if targets is not None:
        rows = lil_matrix((h, size))
        for t in range(h):
            for z in range(n):
                rows[t, z * width + 4 * h + t] = -1 / n
            rows[t, n * width + t] = -1
            reserve_obj[n * width + t] = 1
        upper = vstack([upper, rows.tocsr()], format="csr")
        rhs = np.r_[rhs, -targets]
        second = linprog(
            reserve_obj,
            A_eq=a,
            b_eq=b,
            A_ub=upper,
            b_ub=rhs,
            bounds=bounds,
            method="highs-ds",
            options={"time_limit": 10.0},
        )
        if not second.success:
            raise RuntimeError(second.message)
        upper = vstack([upper, csr_matrix(reserve_obj[None])], format="csr")
        rhs = np.r_[rhs, second.fun + 1e-5]
    last = linprog(
        wear,
        A_eq=a,
        b_eq=b,
        A_ub=upper,
        b_ub=rhs,
        bounds=bounds,
        method="highs-ds",
        options={"time_limit": 10.0},
    )
    if not last.success:
        raise RuntimeError(last.message)
    return float(last.x[h] - last.x[0])
