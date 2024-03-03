import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from problem10 import hubble_parameter_quintessence

# Parameters
N_i = np.log(1 / 3)  # characteristic initial time
N_f = 0  # characteristic stop time


def lumonisity_integrand(V: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Represents the integral over N of the dimensionless luminosity distance (eq. S)

    arguments:
        V: the quintessence potential function in ["power", "exponential"]

    returns:
        The characteristic time array N
        The integrand values array
    """

    z, h = hubble_parameter_quintessence(V, N_i, N_f)
    N = np.log(1 / (1 + z))
    return N, np.exp(-N) / h


def luminosity_distance_quintessence(V: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the dimensionless luminosity distance H_0d_L/c for the quintessence model
    as a function of the characteristic time: ln(1/3) <= N <= 0

    arguments:
        V: the quintessence potential function in ["power", "exponential"]

    returns:
        The characteristic time array N
        The dimensionless luminosity distance
    """
    N, I = lumonisity_integrand(V)

    return N, cumulative_trapezoid(I, N, initial=0)


def plot_lumosity_distances(
    filename: str = None,
    figsize: tuple[int, int] = (6, 4),
    prnt: bool = True,
) -> None:
    """Plots the dimensionless luminosity distance for the two quintessence models

    arguments:
        filename: the filename to save the plot figure
        figsize: the plot figure size
        prnt: if true, prints the edge values

    returns:
        none
    """

    if not filename:
        filename = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "Figures",
                f"12_luminosity_distances.png",
            )
        )

    plt.figure(figsize=figsize)
    for V in ["power", "exponential"]:
        N, d = luminosity_distance_quintessence(V)
        z = np.flip(np.exp(-N) - 1)  # convert time x-axis to the redshift z
        plt.plot(z, d, label=V)

        if prnt:
            print(
                f"Edge values for dimensionless luminosity distance for {V}-potential:"
            )
            print("z=, d_L=")
            print(z[0], d[0])
            print(z[-1], d[-1])
            print()

    # plt.yscale("log")
    plt.xlabel("$z$")
    plt.ylabel(r"$\frac{H_0}{c}d_L$")
    plt.title("Dimensionless luminosity distance for the quintessence models")
    plt.legend()
    plt.grid()
    plt.savefig(filename)


if __name__ == "__main__":
    plot_lumosity_distances()
