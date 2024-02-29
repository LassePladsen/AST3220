import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from problem9 import density_parameters
from expressions import eos_integral_N


def hubble_parameter(V: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns the characteristic Hubble parameter H/H_0 as a function of the redshift z (eq. 7)

    arguments:
        V: the potential function in ["power", "exponential"]

    returns:
        z: the redshift z
        H: the values of H/H_0 for each z from the z-interval
    """
    # Density parameters
    z, Omega_m, Omega_r, Omega_phi = density_parameters(V)
    Omega_m0 = Omega_m[-1]
    Omega_r0 = Omega_r[-1]
    Omega_phi0 = Omega_phi[-1]

    # Convert redshift z to characteristic time N
    N = np.log(1 / (1 + z))

    H = np.empty_like(z)

    for i, zi in enumerate(z):
        H[i] = np.sqrt(
            Omega_m0 * (1 + zi) ** 3
            + Omega_r0 * (1 + zi) ** 4
            + Omega_phi0 * np.exp(quad(eos_integral_N, N[i], 0)[0])
        )

    return z, H


def plot_hubble_parameter(
    V: str, filename: str = None, figsize: tuple[int, int] = (9, 5), prnt=True
) -> None:
    """Plots the characteristic Hubble parameter H/H_0 as a function of the redshift z

    arguments:
        V: the potential function in ["power", "exponential"]
        filename: the name of the file to save the plot
        figsize: the size of the figure
        prnt: if True, print the value of H/H_0 at the initla and final redshifts

    returns:
        None
    """

    if not filename:
        filename = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "Figures",
                f"10_hubble_parameter_{V}.png",
            )
        )

    z, H = hubble_parameter(V)

    plt.figure(figsize=figsize)
    plt.title(f"Hubble parameter for {V}-potential")
    plt.plot(z, H)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("z")
    plt.ylabel("$H/H_0$")
    plt.grid()
    plt.savefig(filename)

    if prnt:
        print("z=, H/H_0=")
        print(z[-1], H[-1])
        print(z[0], H[0])


if __name__ == "__main__":
    for V in ["power", "exponential"]:
        plot_hubble_parameter(V)
