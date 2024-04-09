from functools import lru_cache
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad

# Variables
T_i = 100e9  # initial temperature [K]
T_f = 0.1e9  # final temperature [K]
n_points = int(1e5)  # number of points in the time array
tau = 1700  # fee neutron decay time [s]
N_eff = 3  # effective number of neutrino species

# Constants, in CGS units
c = 2.9979e8  # speed of light [m/s]
# k = 1.380649  # Boltzmann constant [eV/K]
k = 1.380649e-23  # Boltzmann constant [J/K]
# hbar = 6.582119569e-16  # reduced Planck constant [eV*s]
hbar = 6.62607015e-34  # reduced Planck constant [J*s]
G = 6.67430e-11  # gravitational constant [N*m^2/kg^2]
T_0 = 2.725  # CMB temperature [K]
# m_p = 938.27208816e6  # proton mass [eV/c^2]
# m_n = 939.56542052e6  # neutron mass [eV/c^2]
m_p = 1.67262192369e-27  # proton mass [kg]
m_n = 1.67492749804e-27  # neutron mass [kg]
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


# Directory to save figures
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
    return 1 / (1 + np.exp((m_n - m_p) * c * c / (k * T)))


def Y_p_eq(T: float) -> float:
    """Thermal equilibrium value of relative number density of protons Y_p, equation (17) of the project.

    arguments:
        T: temperature [K]

    returns:
        the thermal equilibrium value of Y_n
    """
    return 1 - Y_n_eq(T)


@lru_cache
def Gamma_np(T_9: float, Q: float = q) -> float:
    """Describes the reaction rate of n -> p, equation (12) of the project.

    arguments:
        T_9: temperature [10^9 K]
        Q: mass difference ratio value, it should be set to q or -q

    returns:
        the reaction rate of n -> p
    """

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


def ode_system(lnT: np.ndarray, Y: np.ndarray, N_species: int = 2) -> list[float]:
    """The right hand side of the coupled ODE system, for Y_i ODE's.

    arguments:
        lnT: logarithmic temperature array
        Y: array with initial values of Y_i like so [Y_n, Y_p, ...]
        N_species: The number of interacting particle species, min=2 max=8

    returns:
        The right hand sides of the ode's
    """

    if not (2 <= N_species <= 8):
        raise ValueError(
            "The number of interacting particle species must be between 2 and 8."
        )

    dY = np.zeros(N_species)
    T = np.exp(lnT)
    T_9 = T * 1e-9

    # Neutrons and protons are always included
    Y_n = Y[0]
    Y_p = Y[1]

    # Change for left hand side of the ODE system (n <-> p)
    LHS_change = Y_p * Gamma_pn(T_9) - Y_n * Gamma_np(T_9)

    # Update neutron and proton ODE's
    dY[0] += LHS_change
    dY[1] -= LHS_change

    if N_species > 2:  # Include deuterium
        ...
        """Y_d = Y[2]  

        # n+p <-> D + gamma 
        Y_np = Y_n * Y_p"""

    if N_species > 3:  # include trituim
        ...

    if N_species > 4:  # include helium-3
        ...

    if N_species > 5:  # include helium-4
        ...

    if N_species > 6:  # include lithium-7
        ...

    if N_species > 7:  # include beryllium-7
        ...

    return -dY / H(T)


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
    Y = np.zeros()
    Y = [Y_n_eq(T_i), Y_p_eq(T_i)]

    # Solve the ODE system
    tol = 1e-12
    sol = solve_ivp(
        ode_system,
        [lnT_i, lnT_f],
        Y,
        method="Radau",
        rtol=tol,
        atol=tol,
        dense_output=True,
    )

    t = np.linspace(sol.t[0], sol.t[-1], n_points)
    return t, sol.sol(t)


def plot_relative_number_densities(
    filename: str = None, figsize: tuple[int, int] = (7, 5)
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
    T = np.exp(lnT)

    plt.figure(figsize=figsize)

    # Plot results
    plt.plot(T, Y_n, "r", label="n")
    plt.plot(T, Y_p, "b", label="p")

    # Plot thermal equilibrium values as dotted lines
    plt.plot(T, Y_n_eq(T), "r:")
    plt.plot(T, Y_p_eq(T), "b:")

    # Plot sum
    plt.plot(T, Y_n + Y_p, "k:", label="sum")

    # Plot settings
    plt.gca().invert_xaxis()  # invert x-axis
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("T [K]")
    plt.ylabel(r"$Y_i$")
    plt.ylim(bottom=1e-3, top=2)
    plt.legend()
    # plt.grid(which="major", axis="x")
    # plt.grid(which="both", axis="y")
    plt.grid()
    plt.title("Relative number densities of neutrons and protons")
    plt.savefig(filename)


if __name__ == "__main__":
    plot_relative_number_densities()
