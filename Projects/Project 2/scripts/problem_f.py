from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad

# Variables
T_i = 100e9  # initial temperature [K]
T_f = 0.1e9  # final temperature [K]
tau = 1700  # fee neutron decay time [s]
N_eff = 3  # effective number of neutrino species

# Constants, in CGS units
c = 2.9979e10  # speed of light [cm/s]
k = 8.617e-5  # Boltzmann constant [eV/K]
hbar = 6.582e-16  # reduced Planck constant [eV*s]
G = 6.674e-8  # gravitational constant [cm^3/g/s^2]
T_0 = 2.725  # CMB temperature [K]
m_p = 938.272e6  # proton mass [eV/c^2]
m_n = 939.565e6  # neutron mass [eV/c^2]
q = 2.53  # mass difference ratio
H_0 = 22.686e-19  # Hubble constant [1/s]
Omega_r0 = (  # radiation density parameter today
    8
    * np.pi**3
    / 45
    * G
    / (H_0 * H_0)
    * (k * T_0) ** 4
    / (hbar**3 * c**5)
    * (1 + N_eff * 7 / 8 * (4 / 11) ** (4 / 3))
)


@lru_cache
def T_nu(T: float) -> float:
    """Neutrino temperature conversion

    arguments:
        T: temperature [K]

    returns:
        the neutrino temperature
    """
    return (4 / 11) ** (1 / 3) * T


@lru_cache
def Y_ni(T: float) -> float:
    """Initial value of Y_n, equation (16) of the project.

    arguments:
        T: temperature [K]

    returns:
        the initial value of Y_n
    """
    return 1 / (1 + np.exp((m_n - m_p) * c * c / (k * T)))


def Y_pi(T: float) -> float:
    """Initial value of Y_p, equation (17) of the project.

    arguments:
        T: temperature [K]

    returns:
        the initial value of Y_n
    """
    return 1 - Y_ni(T)


@lru_cache
def Gamma_np(T: float, Q: float = q) -> float:
    """Describes the reaction rate of n -> p, equation (12) of the project.

    arguments:
        T: temperature [K]
        Q: mass difference ratio value, it should be set to q or -q

    returns:
        the reaction rate of n -> p
    """

    T_9 = T * 1e-9

    Z = 5.93 / T_9
    Z_nu = 5.93 / (T_nu(T_9))

    def I1(x):
        """Integrand of the first integral in equation (12) of the project."""
        return (
            (x + Q)
            * (x + Q)
            * np.sqrt(x * x - 1)
            * x
            / ((1 + np.exp(x * Z)) * (1 + np.exp(Z_nu * -(x + Q))))
        )

    def I2(x):
        """Integrand of the second integral in equation (12) of the project."""
        return (
            (x - Q)
            * (x - Q)
            * np.sqrt(x * x - 1)
            * x
            / ((1 + np.exp(-x * Z)) * (1 + np.exp(Z_nu * (x - Q))))
        )

    return 1 / tau * (quad(I1, 1, np.inf)[0] + quad(I2, 1, np.inf)[0])


def Gamma_pn(T: float) -> float:
    """Describes the reaction rate of p -> n, equation (13) of the project.

    arguments:
        T: temperature [K]

    returns:
        the reaction rate of p -> n
    """
    return Gamma_np(T, -q)


@lru_cache
def a(t: float) -> float:
    """Scale factor as function of time, result from problem d.

    arguments:
        t: time [s]

    returns:
        the scale factor
    """
    return np.sqrt(
        2 * H_0 * np.sqrt(Omega_r0) * t + 1
    )  ############## TODO: not sure if this is correct


@lru_cache
def H(t: float) -> float:
    """Hubble parameter as function of time, equation (14) of the project, using
    a(t) result from problm d.

    arguments:
        t: time [s]

    returns:
        the Hubble parameter
    """
    return H_0 * np.sqrt(Omega_r0) * a(t) ** (-2)


def ode_system(lnT: np.ndarray, X: np.ndarray) -> list[float]:
    """System of the two coupled ODE's from equations (10) and (11) of the project.

    arguments:
        lnT: logarithmic temperature array
        X: array with initial values of [Y_n, Y_p]
        V: the potential function in ["power", "exponential"]

    returns:
        The right hand sides of the ode's [dY_n/d(lnT), dY_p/d(lnT)]
    """

    Y_n, Y_p = X

    dY_n = -1 / H * (Y_p * Gamma_pn - Y_n * Gamma_np)
    dY_p = 1 / H * (Y_n * Gamma_np - Y_p * Gamma_pn)

    return [dY_n, dY_p]
