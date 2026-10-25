import numpy as np
import pytest
from reference_blast.lfp_reference import Lfp_Gr_250AhPrismatic

from microgrid.ageing import LfpState, evaluate_lfp, graphite_potential, update_lfp


def test_published_coefficients_and_state_updates_match_pinned_reference():
    ref = Lfp_Gr_250AhPrismatic()
    state = LfpState()
    for day in range(30):
        t = np.arange(25, dtype=float)
        soc = 0.5 + 0.4 * np.sin(2 * np.pi * t / 24)
        temp = np.full(25, 25 + day % 10)
        q = ref.outputs["q"][-1]
        crate = np.abs(np.diff(soc))
        crate[crate < 0.01] = 0
        stress = {
            "t_secs": t * 3600,
            "delta_t_days": 1.0,
            "delta_efc": np.abs(np.diff(soc)).sum() / 2 * q,
            "TdegK": temp + 273.15,
            "soc": soc,
            "Ua": ref._get_Ua(soc),
            "dod": np.ptp(soc),
            "Crate": np.trapezoid(crate, t[1:]) / 24 * q,
        }
        ref.update_rates(stress)
        ref.update_states(stress)
        ref.update_outputs(stress)
        ours = update_lfp(state, t, soc, temp)
        assert ours["kcal"] == pytest.approx(ref.rates["kcal"][-1], rel=1e-12)
        assert ours["kcyc"] == pytest.approx(ref.rates["kcyc"][-1], rel=1e-12)
        assert state.soh == pytest.approx(ref.outputs["q"][-1], abs=1e-12)
        assert np.allclose(graphite_potential(soc), stress["Ua"])


def test_storage_has_calendar_damage_but_no_cycling():
    result = evaluate_lfp(np.full(49, 0.5))
    assert result["calendar_loss_fraction"] > 0
    assert result["cycling_loss_fraction"] == 0
    assert result["rainflow_efc"] == 0


@pytest.mark.parametrize("temperature", [0, 50])
def test_reject_temperature_extrapolation(temperature):
    with pytest.raises(ValueError, match="range"):
        evaluate_lfp(np.array([0.1, 0.5, 0.9]), temperature_c=temperature)


def test_capacity_feedback_accounts_for_lost_energy_and_respects_limits():
    from microgrid.ageing_dispatch import rule_based_with_capacity_fade
    from microgrid.data import synthetic
    from microgrid.physics import BatteryConfig

    c = BatteryConfig(capacity_kwh=200)
    frame = synthetic(72)
    trace, result = rule_based_with_capacity_fade(frame, c)
    assert 0 < result["final_soh"] < 1
    assert result["capacity_fade_energy_loss_kwh"] >= 0
    assert np.all(trace.energy_after_fade_kwh <= c.capacity_kwh * trace.soh * c.soc_max + 1e-8)
    assert np.all(trace.energy_after_fade_kwh >= c.capacity_kwh * trace.soh * c.soc_min - 1e-8)
    p = trace.battery_kw.to_numpy()
    delta = np.where(p >= 0, -p / c.discharge_efficiency, -p * c.charge_efficiency) * c.dt_hours
    final = c.initial_soc * c.capacity + delta.sum() - trace.capacity_fade_energy_loss_kwh.sum()
    assert final == pytest.approx(trace.energy_after_fade_kwh.iloc[-1])
