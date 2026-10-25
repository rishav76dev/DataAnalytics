"""Published-model LFP sensitivity evaluator, not pack-specific empirical validation.

Equations/parameters adapted from NREL BLAST-Lite Lfp_Gr_250AhPrismatic,
commit b093495b47dc40dd96dba865d91f553619501e94
    Copyright (c) 2023, Alliance for Energy Innovation, LLC.
Redistribution terms and disclaimer are retained in BLAST_LICENSE.txt (BSD-3-Clause).
Experimental data: DOI 10.1016/j.est.2023.109042.
"""

from dataclasses import dataclass

import numpy as np
import rainflow


@dataclass
class LfpState:
    calendar_loss: float = 0.0
    cycling_loss: float = 0.0
    efc: float = 0.0

    @property
    def soh(self):
        return 1.0 - self.calendar_loss - self.cycling_loss


def graphite_potential(soc):
    x = 0.0085 + np.asarray(soc) * (0.78 - 0.0085)
    return (
        0.6379
        + 0.5416 * np.exp(-305.5309 * x)
        + 0.044 * np.tanh(-(x - 0.1958) / 0.1088)
        - 0.1978 * np.tanh((x - 1.0571) / 0.0854)
        - 0.6875 * np.tanh((x + 0.0117) / 0.0529)
        - 0.0175 * np.tanh((x - 0.5692) / 0.0875)
    )


def power_increment(loss, exposure, coefficient, exponent):
    # BLAST-Lite differential update, including exact first-step power law.
    if exposure == 0:
        return 0.0
    if loss == 0:
        return coefficient * exposure**exponent
    return coefficient * exponent * (loss / coefficient) ** ((exponent - 1) / exponent) * exposure


def update_lfp(state, time_hours, soc, temperature_c):
    t, s, temp = (
        np.asarray(time_hours, float),
        np.asarray(soc, float),
        np.asarray(temperature_c, float),
    )
    if t.ndim != 1 or len(t) < 3 or t.shape != s.shape or t.shape != temp.shape:
        raise ValueError("Need equal one-dimensional histories with at least three samples")
    if (
        not all(np.isfinite(a).all() for a in [t, s, temp])
        or (np.diff(t) <= 0).any()
        or ((s < 0) | (s > 1)).any()
    ):
        raise ValueError("Invalid time, SOC or temperature history")
    if (temp < 10).any() or (temp > 45).any():
        raise ValueError("Temperature outside published cycling range [10,45] C")
    if state.soh <= 0.5:
        raise ValueError("Model beyond configured end-of-life applicability")
    crate_series = np.abs(np.diff(s) / np.diff(t))
    if crate_series.max() * state.soh > 0.65 + 1e-9:
        raise ValueError("Conservative <=0.65C rate envelope exceeded")
    duration = t[-1] - t[0]
    # Match upstream nominal-capacity EFC/C-rate scaling at current health.
    efc = np.abs(np.diff(s)).sum() / 2 * state.soh
    filtered = crate_series.copy()
    filtered[filtered < 0.01] = 0
    crate = np.trapezoid(filtered, t[1:]) / duration * state.soh
    dod = float(np.ptp(s))
    kelvin = temp + 273.15
    kcal = (
        np.trapezoid(
            8.37e4 * np.exp(-5210 / kelvin) * np.exp(-3560 * graphite_potential(s) / kelvin), t
        )
        / duration
    )
    kcyc = (
        np.trapezoid(
            (4.38e-8 + 1.55e-8 * dod + 1.68e-7 * crate)
            * (np.exp(2190 / kelvin) + np.exp(-1.55e5 / kelvin)),
            t,
        )
        / duration
    )
    dc = power_increment(state.calendar_loss, duration / 24, kcal, 0.526)
    dy = power_increment(state.cycling_loss, efc, kcyc, 0.828)
    state.calendar_loss += float(dc)
    state.cycling_loss += float(dy)
    state.efc += float(efc)
    return {
        "calendar_increment": float(dc),
        "cycling_increment": float(dy),
        "kcal": float(kcal),
        "kcyc": float(kcyc),
        "efc_increment": float(efc),
        "dod": dod,
        "crate": float(crate),
    }


def evaluate_lfp(soc, dt_hours=1.0, temperature_c=25.0, chunk_hours=24):
    """Piecewise daily offline ageing
    dispatch SOH remains fixed unless coupled explicitly."""
    s = np.asarray(soc, float)
    if len(s) < 3 or dt_hours <= 0 or chunk_hours < 2 * dt_hours:
        raise ValueError("Insufficient history or invalid timestep/chunk duration")
    temp = np.broadcast_to(np.asarray(temperature_c, float), s.shape)
    state = LfpState()
    step = max(2, int(chunk_hours / dt_hours))
    start = 0
    windows = []
    while start < len(s) - 1:
        stop = min(start + step, len(s) - 1)
        if len(s) - 1 - stop == 1:
            stop += 1
        windows.append(
            update_lfp(
                state,
                np.arange(stop - start + 1) * dt_hours,
                s[start : stop + 1],
                temp[start : stop + 1],
            )
        )
        start = stop
    cycles = list(rainflow.extract_cycles(s))
    return {
        "model": "BLAST-Lite LFP-Gr 250Ah published-data model",
        "source_doi": "10.1016/j.est.2023.109042",
        "source_commit": "b093495b47dc40dd96dba865d91f553619501e94",
        "status": "published_parameter_model_numerically_verified_not_pack_validated",
        "soh": state.soh,
        "calendar_loss_fraction": state.calendar_loss,
        "cycling_loss_fraction": state.cycling_loss,
        "total_loss_fraction": 1 - state.soh,
        "throughput_efc_soh_adjusted": state.efc,
        "rainflow_efc": float(sum(depth * count for depth, _, count, _, _ in cycles)),
        "temperature_assumption_c": float(temp.mean()),
        "update_chunk_hours": chunk_hours,
        "dispatch_soh_feedback": False,
        "windows": windows,
        "limitation": "Daily DOD below fitted 0.8–1 range is extrapolation; cell model is not validated for this pack.",
    }
