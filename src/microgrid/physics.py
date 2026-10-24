from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 100.0
    power_kw: float = 50.0
    soc_min: float = 0.1
    soc_max: float = 0.9
    initial_soc: float = 0.5
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    soh: float = 1.0
    dt_hours: float = 1.0

    def __post_init__(self):
        values = list(self.__dict__.values())
        if not all(np.isfinite(values)):
            raise ValueError("Battery parameters must be finite")
        if not (self.capacity_kwh > 0 and self.power_kw > 0 and self.dt_hours > 0):
            raise ValueError("Capacity, power, and timestep must be positive")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ValueError("Invalid SOC limits")
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError("Initial SOC outside limits")
        if not (
            0 < self.soh <= 1
            and 0 < self.charge_efficiency <= 1
            and 0 < self.discharge_efficiency <= 1
        ):
            raise ValueError("Invalid SOH or efficiency")

    @property
    def capacity(self):
        return self.capacity_kwh * self.soh

    @property
    def minimum(self):
        return self.capacity * self.soc_min

    @property
    def maximum(self):
        return self.capacity * self.soc_max


class Battery:
    """Positive power discharges at the AC bus; negative power charges."""

    def __init__(self, config=BatteryConfig()):
        self.config = config
        self.energy = config.initial_soc * config.capacity

    @property
    def soc(self):
        return self.energy / self.config.capacity

    def project(self, requested_kw, net_load_kw):
        c = self.config
        if not np.isfinite([requested_kw, net_load_kw]).all():
            raise ValueError("Dispatch must be finite")
        discharge = min(
            c.power_kw,
            (self.energy - c.minimum) * c.discharge_efficiency / c.dt_hours,
            max(net_load_kw, 0),
        )
        charge = min(
            c.power_kw,
            (c.maximum - self.energy) / (c.charge_efficiency * c.dt_hours),
            max(-net_load_kw, 0),
        )
        return float(np.clip(requested_kw, -max(charge, 0), max(discharge, 0)))

    def step(self, requested_kw, pv_kw, wind_kw, load_kw):
        if not np.isfinite([pv_kw, wind_kw, load_kw]).all() or min(pv_kw, wind_kw, load_kw) < 0:
            raise ValueError("Generation and load must be finite and nonnegative")
        c = self.config
        net = load_kw - pv_kw - wind_kw
        power = self.project(requested_kw, net)
        before = self.energy
        self.energy += (
            -power / c.discharge_efficiency if power >= 0 else -power * c.charge_efficiency
        ) * c.dt_hours
        self.energy = float(np.clip(self.energy, c.minimum, c.maximum))
        unmet = max(net - power, 0.0)
        curtailed = max(power - net, 0.0)
        return dict(
            battery_kw=power,
            energy_kwh=self.energy,
            soc=self.soc,
            unserved_kw=unmet,
            curtailment_kw=curtailed,
            throughput_kwh=abs(self.energy - before),
            load_kw=load_kw,
            pv_kw=pv_kw,
            wind_kw=wind_kw,
        )
