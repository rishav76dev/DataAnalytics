"""Published-model capacity-feedback sensitivity for rule-based dispatch."""

from dataclasses import replace

import numpy as np
import pandas as pd

from .ageing import LfpState, update_lfp
from .data import CHANNELS, audit
from .evaluation import reliability
from .physics import Battery, BatteryConfig


def rule_based_with_capacity_fade(frame, config=BatteryConfig(), temperature_c=25.0):
    """Daily cell-model fade; NOT calibrated pack lifetime or an RL comparison.

    Preserve SOC when applying capacity fade. Log removed stored energy separately
    from dispatch losses, curtailment and unmet demand. No hidden energy injection.
    """
    audit(frame)
    battery = Battery(config)
    state = LfpState()
    rows = []
    soc_window = [battery.soc]
    for i, x in enumerate(frame[CHANNELS].to_numpy()):
        row = battery.step(float(x[2] - x[0] - x[1]), *x)
        soc_window.append(battery.soc)
        removed = 0.0
        if len(soc_window) == 25 or i == len(frame) - 1:
            history = np.array(soc_window)
            times = np.arange(len(history)) * config.dt_hours
            if len(history) == 2:
                history = np.array([history[0], history.mean(), history[1]])
                times = np.array([0, 0.5, 1]) * config.dt_hours
            update_lfp(state, times, history, np.full(len(history), temperature_c))
            old_capacity = battery.config.capacity
            new_config = replace(config, soh=config.soh * state.soh)
            new_energy = battery.energy * new_config.capacity / old_capacity
            removed = battery.energy - new_energy
            battery.config = new_config
            battery.energy = new_energy
            soc_window = [battery.soc]
        row.update(
            {
                "timestamp": frame.timestamp.iloc[i],
                "soh": battery.config.soh,
                "energy_after_fade_kwh": battery.energy,
                "capacity_fade_energy_loss_kwh": removed,
            }
        )
        rows.append(row)
    trace = pd.DataFrame(rows)
    metrics = reliability(trace, config.dt_hours)
    metrics.update(
        {
            "model": "published BLAST-Lite LFP-Gr 250Ah capacity-feedback sensitivity",
            "temperature_c": temperature_c,
            "final_soh": battery.config.soh,
            "capacity_fade_energy_loss_kwh": float(trace.capacity_fade_energy_loss_kwh.sum()),
            "calendar_loss_fraction": state.calendar_loss,
            "cycling_loss_fraction": state.cycling_loss,
            "applicability": "cell parameter model, not pack validated; national proxy case; conservative constant battery temperature",
            "state_update": "SOC preserved as capacity shrinks; unavailable stored energy recorded separately",
        }
    )
    return trace, metrics
