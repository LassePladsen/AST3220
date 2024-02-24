import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from expressions import xi, ode_system


def solve_ode_system(V: str) -> list[np.ndarray, np.ndarray]:
    """Solves the ode system of the equations of motion for x1, x2, x3, and lambda

    arguments:
        V: the potential function in ["power", "exponential"]

    returns:
        the time array and the solution array
    """

    ## Initial conditions and variables
    N_i = np.log(1 / (1 + 2e7))  # characteristic initial time
    N_f = 0  # characteristic stop time

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

    sol = solve_ivp(
        ode_system,
        [N_i, N_f],
        [x1_i, x2_i, x3_i, lmbda_i],
        args=(V,),
        rtol=1e-8,
        atol=1e-8,
    )

    return sol.t, sol.y


def plot_density_parameters(V: str, filename=None, figsize=(9, 5)) -> None:
    """Plots the characteristic density parameters (Omega_i) for matter, radiation,
    and the quintessence field as functions of the redshift, in the same figure

    argfuments:
        V: the potential function in ["power", "exponential"]

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

    N, y = solve_ode_system(V)
    z = np.exp(-N) - 1  # redshift
    # z = N

    x1, x2, x3, _ = y

    # Density parameters
    Omega_m = 1 - x1**2 - x2**2 - x3**2
    Omega_r = x3**2
    Omega_phi = x1**2 + x2**2

    # Plot seperately in the same figure
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    ax[0].plot(z, Omega_m)
    ax[1].plot(z, Omega_r)
    ax[2].plot(z, Omega_phi)

    # Titles
    ax[0].set_title(r"Matter $\Omega_m$")
    ax[1].set_title(r"Radiation $\Omega_r$")
    ax[2].set_title(r"Quintessence field $\Omega_{\phi}$")

    # Shared labels and figure title
    fig.supxlabel("z")
    fig.supylabel(r"$\Omega$")
    fig.suptitle(f"Density parameters of {V}-potential")

    plt.savefig(filename)


def plot_eos_parameter(V: str) -> None:
    """Plots the quintessence field equation of state parameter w_phi as
    a function of the redshift

    arguments:
        V: the potential function in ["power", "exponential"]

    returns:
        None
    """
    ...


if __name__ == "__main__":
    # Create figures for both potentials
    for V in ["power", "exponential"]:
        plot_density_parameters(V)
        plot_eos_parameter(V)
