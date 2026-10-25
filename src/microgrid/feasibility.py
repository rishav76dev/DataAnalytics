"""Full-episode sparse LP adequacy diagnostic; primary objective is EENS only."""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .data import CHANNELS, audit
from .evaluation import dispatch, reliability
from .physics import BatteryConfig


def perfect_foresight_episode(frame, config=BatteryConfig(), terminal="minimum"):
    audit(frame)
    if terminal not in {"minimum", "initial"}:
        raise ValueError("terminal must be minimum or initial")
    c, n = config, len(frame)
    net = (frame.load_kw - frame.pv_kw - frame.wind_kw).to_numpy()
    # [charge, discharge, shed, curtail, end_energy], each length n.
    a = lil_matrix((2 * n, 5 * n))
    b = np.zeros(2 * n)
    bounds = (
        [(0, min(c.power_kw, max(-v, 0))) for v in net]
        + [(0, min(c.power_kw, max(v, 0))) for v in net]
        + [(0, max(v, 0)) for v in net]
        + [(0, max(-v, 0)) for v in net]
        + [(c.minimum, c.maximum)] * n
    )
    terminal_floor = c.minimum if terminal == "minimum" else c.initial_soc * c.capacity
    bounds[-1] = (terminal_floor, c.maximum)
    for t in range(n):
        a[t, t], a[t, n + t], a[t, 2 * n + t], a[t, 3 * n + t] = -1, 1, 1, -1
        b[t] = net[t]
        a[n + t, 4 * n + t] = 1
        a[n + t, t] = -c.charge_efficiency * c.dt_hours
        a[n + t, n + t] = c.dt_hours / c.discharge_efficiency
        if t:
            a[n + t, 4 * n + t - 1] = -1
        else:
            b[n + t] = c.initial_soc * c.capacity
    objective = np.zeros(5 * n)
    objective[2 * n : 3 * n] = c.dt_hours
    result = linprog(objective, A_eq=a.tocsr(), b_eq=b, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Adequacy LP failed: {result.message}")
    x = result.x
    trace = frame[["timestamp", *CHANNELS]].copy()
    trace["battery_kw"] = x[n : 2 * n] - x[:n]
    trace["unserved_kw"], trace["curtailment_kw"] = x[2 * n : 3 * n], x[3 * n : 4 * n]
    trace["energy_kwh"] = x[4 * n :]
    trace["soc"] = trace.energy_kwh / c.capacity
    trace["throughput_kwh"] = np.abs(np.diff([c.initial_soc * c.capacity, *trace.energy_kwh]))
    report = reliability(trace, c.dt_hours)
    report.update(
        {
            "reference": "non_deployable_complete_episode_EENS_optimum",
            "terminal_policy": terminal,
            "terminal_energy_kwh": float(trace.energy_kwh.iloc[-1]),
            "minimum_power_shortage_bound_kwh": float(
                np.maximum(net - c.power_kw, 0).sum() * c.dt_hours
            ),
            "renewable_kwh": float((frame.pv_kw + frame.wind_kw).sum() * c.dt_hours),
            "max_balance_residual_kw": float(np.abs(a @ x - b).max()),
            "target_lpsp": 0.01,
            "target_possible_under_foresight": report["lpsp_energy"] <= 0.01,
        }
    )
    return trace, report


def sizing_study(
    frame,
    capacities=(100, 200, 400),
    powers=(25, 50, 100),
    renewable_scales=(0.75, 1, 1.25, 1.5),
    initial_socs=(0.5,),
    terminals=("minimum",),
):
    rows = []
    for capacity in capacities:
        for power in powers:
            for scale in renewable_scales:
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("Renewable scale must be positive")
                selected = frame.copy()
                selected[["pv_kw", "wind_kw"]] *= scale
                for initial in initial_socs:
                    c = BatteryConfig(capacity_kwh=capacity, power_kw=power, initial_soc=initial)
                    _, rule = dispatch(selected, lambda t, e, x: x[2] - x[0] - x[1], c)
                    for terminal in terminals:
                        _, oracle = perfect_foresight_episode(selected, c, terminal)
                        rows.append(
                            {
                                "capacity_kwh": capacity,
                                "power_kw": power,
                                "renewable_scale": scale,
                                "initial_soc": initial,
                                "terminal": terminal,
                                "rule_based_eens_kwh": rule["eens_kwh"],
                                **oracle,
                            }
                        )
    return pd.DataFrame(rows)
