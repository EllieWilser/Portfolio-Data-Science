"""Physics part of the hybrid home thermal model.

A 1R1C ("one resistor, one capacitor") model of a house:

    C * dT_in/dt = (T_out - T_in) / R + Q_hvac + a_sol * I_sol

    T_in   indoor air temperature [degC]
    T_out  outdoor air temperature [degC]
    R      envelope thermal resistance [degC/kW]
    C      thermal capacitance of air + furniture + walls [kWh/degC]
    Q_hvac heating (+) or cooling (-) power delivered [kW]
    I_sol  global horizontal solar irradiance [kW/m2]
    a_sol  effective solar aperture [m2]

The neural-network residual (step 4 of the plan) is added on top of this later.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def simulate_1r1c(
    t_out: np.ndarray,
    q_hvac: np.ndarray,
    i_sol: np.ndarray,
    R: float,
    C: float,
    a_sol: float,
    t_in0: float,
    dt_h: float = 0.25,
) -> np.ndarray:
    """Simulate indoor temperature with a forward-Euler step of size dt_h hours."""
    n = len(t_out)
    t_in = np.empty(n)
    t_in[0] = t_in0
    for k in range(n - 1):
        dTdt = ((t_out[k] - t_in[k]) / R + q_hvac[k] + a_sol * i_sol[k]) / C
        t_in[k + 1] = t_in[k] + dt_h * dTdt
    return t_in


def fit_1r1c(
    t_in_obs: np.ndarray,
    t_out: np.ndarray,
    q_hvac: np.ndarray,
    i_sol: np.ndarray,
    dt_h: float = 0.25,
    x0: tuple[float, float, float] = (5.0, 5.0, 1.0),
) -> dict:
    """Fit R, C and a_sol by minimising simulated-vs-observed indoor temperature error."""

    def residuals(params):
        R, C, a_sol = params
        sim = simulate_1r1c(t_out, q_hvac, i_sol, R, C, a_sol, t_in_obs[0], dt_h)
        return sim - t_in_obs

    res = least_squares(residuals, x0=x0, bounds=([0.1, 0.1, 0.0], [100, 100, 50]))
    R, C, a_sol = res.x
    rmse = float(np.sqrt(np.mean(res.fun**2)))
    return {"R": R, "C": C, "a_sol": a_sol, "rmse": rmse}


def synthetic_house(days: int = 14, dt_h: float = 0.25, seed: int = 0) -> dict:
    """Make a toy dataset with known R, C, a_sol so we can check the fit recovers them.

    Uses sinusoidal weather for now; step 1 of the plan swaps in real NOAA/NSRDB weather.
    Includes a hidden 'internal gains' signal (occupancy) that the 1R1C model does not
    know about: this is exactly what the neural-network residual should learn later.
    """
    rng = np.random.default_rng(seed)
    n = int(days * 24 / dt_h)
    hours = np.arange(n) * dt_h
    t_out = 2 + 6 * np.sin(2 * np.pi * (hours - 9) / 24) + rng.normal(0, 0.5, n)
    i_sol = np.clip(0.8 * np.sin(2 * np.pi * (hours - 6) / 24), 0, None)

    # simple thermostat: heat at 3 kW when below 20 degC (decided from previous step)
    true = {"R": 8.0, "C": 3.0, "a_sol": 2.0}
    occupancy_gain = 0.4 * ((hours % 24 > 17) | (hours % 24 < 7))  # evenings and nights
    t_in = np.empty(n)
    q_hvac = np.zeros(n)
    t_in[0] = 20.0
    for k in range(n - 1):
        q_hvac[k] = 3.0 if t_in[k] < 20.0 else 0.0
        dTdt = (
            (t_out[k] - t_in[k]) / true["R"]
            + q_hvac[k]
            + true["a_sol"] * i_sol[k]
            + occupancy_gain[k]
        ) / true["C"]
        t_in[k + 1] = t_in[k] + dt_h * dTdt
    t_in += rng.normal(0, 0.05, n)  # sensor noise

    return {
        "hours": hours,
        "t_out": t_out,
        "i_sol": i_sol,
        "q_hvac": q_hvac,
        "t_in": t_in,
        "true_params": true,
        "dt_h": dt_h,
    }


if __name__ == "__main__":
    d = synthetic_house()
    fit = fit_1r1c(d["t_in"], d["t_out"], d["q_hvac"], d["i_sol"], d["dt_h"])
    print("true:", d["true_params"])
    print("fit: ", {k: round(float(v), 3) for k, v in fit.items()})
