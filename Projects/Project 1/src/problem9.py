import os
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Constants
G = 6.6743e-11  # Newton gravitational constant [m^3/(kg s^2)]
alpha = 1
xi = 3 / 2

# Differential equation parameters
N_i = np.log(1 / (1 + 2e7))  # characteristic initial time
N_f = 0  # characteristic stop time


def gamma(V: str) -> float:
    """Describes Gamma(phi), equation 18 of the project, which depends on
    the potential

    arguments:
        V: the potential function in ["power", "exponential"]

    returns:
        the value of Gamma (eq. 18)
    """
    if V.lower() == "power":
        return (alpha + 1) / alpha
    elif V.lower() == "exponential":
        return 1
    else:
        raise ValueError("V-string value not recognized")


def dlambda(X: np.ndarray, V: str) -> float:
    """Describes dlambda/dN, equation 22 of the project, which depends on the potential.

    arguments:
        X: array with values of [x1, x2, x3, lmbda]
        V: the potential function: ["power", "exponential"]

    returns:
        the right hand side of dlambda/dN (eq. 22)
    """

    x1, _, _, lmbda = X

    return -np.sqrt(6) * lmbda**2 * (gamma(V) - 1) * x1


def ode_system(N: np.ndarray, X: np.ndarray, V: str) -> list[float]:
    """System of the four coupled ODE's from expression 19-22 of the project.

    arguments:
        N: characteristic time array (eq. 15)
        X: array with values of [x1, x2, x3, lmbda]
        V: the potential function in ["power", "exponential"]

    returns:
        array with the right hand sides of the ode's [dx1/dN, dx2/dN, dx3/dN, dlmbda/dN]
    """

    x1, x2, x3, lmbda = X

    hubble_expression = -0.5 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
    dx1 = -3 * x1 + np.sqrt(6) / 2 * lmbda * x2**2 - x1 * hubble_expression
    dx2 = -np.sqrt(6) / 2 * lmbda * x1 * x2 - x2 * hubble_expression
    dx3 = -2 * x3 - x3 * hubble_expression
    dlmbda = dlambda(X, V)

    return [dx1, dx2, dx3, dlmbda]


def solve_ode_system(V: str, N_i: float, N_f: float) -> tuple[np.ndarray, np.ndarray]:
    """Solves the ode system of the equations of motion for x1, x2, x3, and lambda

    arguments:
        V: the potential function in ["power", "exponential"]
        N_i: the characteristic initial time
        N_f: the characteristic stop time

    returns:
        the time array and the solution array
    """

    # Parameters
    if V.lower() == "power":
        x1_i = 5e-5
        x2_i = 1e-8
        x3_i = 0.9999
        lmbda_i = 1e9
    elif V.lower() == "exponential":
        x1_i = 0
        x2_i = 5e-13
        x3_i = 0.9999
        lmbda_i = xi
    else:
        raise ValueError("V-string value not recognized")

    n_points = int(1e6)
    tol = 1e-8
    sol = solve_ivp(
        ode_system,
        [N_i, N_f],
        [x1_i, x2_i, x3_i, lmbda_i],
        args=(V,),
        rtol=tol,
        atol=tol,
        t_eval=np.linspace(N_i, N_f, n_points),
    )

    return sol.t, sol.y


def density_parameters(
    V: str,
    N_i: float,
    N_f: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns the characteristic density parameters (Omega_i) for matter, radiation,
    and the quintessence field as functions of the redshift

    arguments:
        V: the potential function in ["power", "exponential"]
        N_i: the characteristic initial time
        N_f: the characteristic stop time

    returns:
        the redshift array and the density parameters arrays
    """

    N, y = solve_ode_system(V, N_i, N_f)
    z = np.exp(-N) - 1  # convert time x-axis to the redshift z

    x1, x2, x3, _ = y

    # Density parameters
    Omega_m = 1 - x1**2 - x2**2 - x3**2
    Omega_r = x3**2
    Omega_phi = x1**2 + x2**2

    return z, Omega_m, Omega_r, Omega_phi


def plot_density_parameters(
    V: str,
    filename: str = None,
    figsize: tuple[int, int] = (9, 5),
    prnt: bool = True,
) -> None:
    """Plots the characteristic density parameters (Omega_i) for matter, radiation,
    and the quintessence field as functions of the redshift, in the same figure

    argfuments:
        V: the potential function in ["power", "exponential"]
        filename: the filename to save the plot figure
        figsize: the plot figure size
        prnt: if true, prints today's values
    returns:
        None
    """

    if not filename:
        filename = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "Figures",
                f"9_density_parameters_{V}.png",
            )
        )

    z, Omega_m, Omega_r, Omega_phi = density_parameters(V, N_i, N_f)

    # Plot in the same figure
    plt.figure(figsize=figsize)
    plt.grid()
    plt.xscale("log")
    plt.plot(z, Omega_m, label=r"Matter $\Omega_m$")
    plt.plot(z, Omega_r, label=r"Radiation $\Omega_r$")
    plt.plot(z, Omega_phi, label=r"Quintessence field $\Omega_{\phi}$")
    plt.legend()

    # Labels and figure title
    plt.xlabel("z")
    plt.ylabel(r"$\Omega$")
    plt.title(f"Density parameters for {V}-potential")

    plt.savefig(filename)

    if prnt:
        print(
            f"Today's values of the density parameters ({V}-potential):"
            f"\nOmega_m0 = {Omega_m[-1]}"
            f"\nOmega_r0 = {Omega_r[-1]}"
            f"\nOmega_phi0 = {Omega_phi[-1]}\n"
        )


def eos_parameter(V: str, N_i: float, N_f: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns the quintessence field equation of state parameter omega_phi as
    a function of the redshift

    arguments:
        V: the potential function in ["power", "exponential"]
        N_i: the characteristic initial time
        N_f: the characteristic stop time

    returns:
        the redshift array and the eos parameter array
    """

    N, y = solve_ode_system(V, N_i, N_f)
    z = np.exp(-N) - 1  # convert time x-axis to the redshift z

    x1, x2, _, _ = y

    # The eos parameter
    omega_phi = (x1**2 - x2**2) / (x1**2 + x2**2)

    return z, omega_phi


def plot_eos_parameter(
    V: str,
    filename: str = None,
    figsize: tuple[int, int] = (9, 5),
    prnt: bool = True,
) -> None:
    """Plots the quintessence field equation of state parameter w_phi as
    a function of the redshift

    arguments:
        V: the potential function in ["power", "exponential"]
        filename: the filename to save the plot figure
        figsize: the plot figure size
        prnt: if true, prints today's values

    returns:
        None
    """
    if not filename:
        filename = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "Figures",
                f"9_eos_parameter_{V}.png",
            )
        )

    z, omega_phi = eos_parameter(V, N_i, N_f)

    # Plot in the same figure
    plt.figure(figsize=figsize)
    plt.grid()
    plt.xscale("log")
    plt.plot(z, omega_phi)

    # Labels and figure title
    plt.xlabel("z")
    plt.ylabel(r"$\Omega$")
    plt.title(f"EoS parameter $\omega_\phi$ for {V}-potential")

    plt.savefig(filename)

    if prnt:
        print(
            f"Today's value of the EoS parameter ({V}-potential):"
            f"\nomega_phi0 = {omega_phi[-1]}\n"
        )


if __name__ == "__main__":
    # Create figures for both potentials
    figsize = (7, 5)
    for V in ["power", "exponential"]:
        plot_density_parameters(V, figsize=figsize)
        plot_eos_parameter(V, figsize=figsize)
