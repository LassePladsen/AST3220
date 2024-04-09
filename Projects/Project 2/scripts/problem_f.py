from functools import lru_cache
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad

# Variables
T_i = 100e9  # initial temperature [K]
T_f = 0.1e9  # final temperature [K]
n_points = 1000  # number of points in the time array
tau = 1700  # fee neutron decay time [s]
N_eff = 3  # effective number of neutrino species

# Constants, in CGS units
c = 2.9979e8  # speed of light [m/s]
k = 1.380649  # Boltzmann constant [eV/K]
hbar = 6.582119569e-16  # reduced Planck constant [eV*s]
G = 6.67430e-11  # gravitational constant [N*m^2/kg^2]
T_0 = 2.725  # CMB temperature [K]
m_p = 938.27208816e6  # proton mass [eV/c^2]
m_n = 939.56542052e6  # neutron mass [eV/c^2]
q = 2.53  # mass difference ratio [(m_n - m_p) / m_e]
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


# Directort to save figures
FIGURES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))

if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)


@lru_cache
def T_nu(T: float) -> float:
    """Neutrino temperature conversion

    arguments:
        T: temperature [K]

    returns:
        the neutrino temperature
    """
    return (4 / 11) ** (1 / 3) * T


def Y_n_eq(T: float) -> float:
    """Thermal equilibrium value of relative number density of neutrons Y_n, equation (16) of the project.

    arguments:
        T: temperature [K]

    returns:
        the thermal equilibrium value of Y_n
    """
    return 1 / (1 + np.exp((m_n - m_p) / (k * T)))


def Y_p_eq(T: float) -> float:
    """Thermal equilibrium value of relative number density of protons Y_p, equation (17) of the project.

    arguments:
        T: temperature [K]

    returns:
        the thermal equilibrium value of Y_n
    """
    return 1 - Y_n_eq(T)


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
def a(T: float) -> float:
    """Scale factor as function of temperature

    arguments:
        T: temperature [K]

    returns:
        the scale factor
    """
    return T_0 / T


@lru_cache
def H(T: float) -> float:
    """Hubble parameter as function of temperature, equation (14) of the project

    arguments:
        t: temperature [K]

    returns:
        the Hubble parameter
    """
    return H_0 * np.sqrt(Omega_r0) * a(T) ** (-2)


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

    dY_n = (
        -1
        / H(np.exp(lnT))
        * (Y_p * Gamma_pn(np.exp(lnT)) - Y_n * Gamma_np(np.exp(lnT)))
    )
    dY_p = (
        -1
        / H(np.exp(lnT))
        * (Y_n * Gamma_np(np.exp(lnT)) - Y_p * Gamma_pn(np.exp(lnT)))
    )

    return [dY_n, dY_p]


def solve_ode_system() -> tuple[np.ndarray, np.ndarray]:
    """Solves the ode system of the equations of motion for Y_n and Y_p

    arguments:
        None

    returns:
        the time array and the solution array
    """

    # Parameters
    lnT_i = np.log(T_i)
    lnT_f = np.log(T_f)

    # Initial conditions
    X_i = [Y_n_eq(T_i), Y_p_eq(T_i)]

    # Solve the ODE system
    tol = 1e-12
    sol = solve_ivp(
        ode_system,
        [lnT_i, lnT_f],
        X_i,
        method="Radau",
        rtol=tol,
        atol=tol,
        dense_output=True,
    )

    t = np.linspace(sol.t[0], sol.t[-1], n_points)
    return t, sol.sol(t)


def plot_relative_number_densities(
    filename: str = None, figsize: tuple[int, int] = (9, 5)
) -> None:
    """Plots the relative number densities of neutrons and protons as a function of logarithmic temperature ln(T)

    arguments:
        filename: the filename to save the plot figure
        figsize: the plot figure size

    returns:
        None
    """

    # Make default filename
    if not filename:
        filename = os.path.join(
            FIGURES_DIR,
            f"f_relative_number_densities.png",
        )

    lnT, X = solve_ode_system()
    Y_n, Y_p = X

    plt.figure(figsize=figsize)

    # Plot results
    plt.plot(lnT, Y_n, "r", label="n")
    plt.plot(lnT, Y_p, "b", label="p")

    # Plot thermal equilibrium values as dotted line
    # print(Y_n_eq(np.exp(lnT)))
    plt.plot(lnT, Y_n_eq(np.exp(lnT)), "r--")
    plt.plot(lnT, Y_p_eq(np.exp(lnT)), "b--")

    # Plot settings
    plt.xlabel(r"$\ln(T)$")
    plt.ylabel(r"$Y_i$")
    plt.legend()
    plt.grid()
    plt.title("Relative number densities of neutrons and protons")
    plt.savefig(filename)


if __name__ == "__main__":
    print(Y_n_eq(np.array([1e5, 1e7, 1e10])))
    # plot_relative_number_densities()
