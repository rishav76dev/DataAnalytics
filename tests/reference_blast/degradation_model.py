# Minimal reference excerpt from BLAST-Lite, commit b093495b47dc40dd96dba865d91f553619501e94.
# Copyright (c) 2023, Alliance for Energy Innovation, LLC. See LICENSE.
# Only upstream power-state and graphite-potential methods retained for regression tests.
import numpy as np

class BatteryDegradationModel:
    @staticmethod
    def _update_power_state(y0: float, dx: float, k: float, p: float):
        """
        Update time-varying power state
        Trajectory equation: y = k*x^p
        State equation: dy = k*p*(k/y)^((1-p)/p)

        Args:
            y0 (float): current state value
            dx (float): change of the independent variable over this timestep
            k (float): coefficient of x^p
            p (float): power of x

        Return:
            dy (float): the change of the state over this timestep
        """
        if y0 == 0:
            if dx == 0:
                dydx = 0
            else:
                y0 = k * (dx**p)
                dydx = y0 / dx
        else:
            if dx == 0:
                dydx = 0
            else:
                dydx = k * p * ((y0 / k) ** ((p - 1) / p))
        return dydx * dx
    @staticmethod
    def _get_Ua(soc: np.ndarray):
        """
        Calculate Ua from SOC via lithiation fraction.

        Args:
            soc (np.ndarray)    SOC profile

        Return:
            Ua
        """
        # Calculate lithiation fraction from soc
        def get_Xa(soc):
            return 8.5 * 10**-3 + soc * (0.78 - 8.5 * 10**-3)

        Xa = get_Xa(soc)
        # Calculate Ua from lithiation fraction
        return (
            0.6379
            + 0.5416 * np.exp(-305.5309 * Xa)
            + 0.044 * np.tanh(-1 * (Xa - 0.1958) / 0.1088)
            - 0.1978 * np.tanh((Xa - 1.0571) / 0.0854)
            - 0.6875 * np.tanh((Xa + 0.0117) / 0.0529)
            - 0.0175 * np.tanh((Xa - 0.5692) / 0.0875)
        )
