import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from problem10 import hubble_parameter_quintessence


def lumonisity_integrand_quintessence(
    V: str, N_i: float, N_f: float, n_points: int = int(1e6)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Represents the integral over z of the dimensionless luminosity distance (eq. S)
    for the quintessence model

    arguments:
        V: the quintessence potential function in ["power", "exponential"]
        N_i: the characteristic initial time
        N_f: the characteristic stop time
        n_points: the number of points to evaluate the integral

    returns:
        z: The redshift array
        I: The integrand values array
    """

    z, H = hubble_parameter_quintessence(V, N_i, N_f, n_points)
    z = np.flip(z)
    return z, 1 / H


def luminosity_distance_quintessence(
    V: str, N_i: float, N_f: float, n_points: int = int(1e6)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the dimensionless luminosity distance H_0 d_L/c for the quintessence model
    as a function of the redshift

    arguments:
        V: the quintessence potential function in ["power", "exponential"]
        N_i: the characteristic initial time
        N_f: the characteristic stop time
        n_points: the number of points to evaluate the integral

    returns:
        z: The redshift array
        d: The dimensionless luminosity distances array
    """
    z, I = lumonisity_integrand_quintessence(V, N_i, N_f, n_points)

    return z, (1 + z) * cumulative_trapezoid(I, z, initial=0)


def plot_lumosity_distances(
    N_i: float,
    N_f: float,
    filename: str = None,
    figsize: tuple[int, int] = (6, 4),
    prnt: bool = True,
) -> None:
    """Plots the dimensionless luminosity distance for the two quintessence models

    arguments:
        N_i: the characteristic initial time
        N_f: the characteristic stop time
        filename: the filename to save the plot figure
        figsize: the plot figure size
        prnt: if true, prints the edge values

    returns:
        none
    """

    # Default filename
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

    # The two quintessence models
    for V in ["power", "exponential"]:
        z, d = luminosity_distance_quintessence(V, N_i, N_f)
        plt.plot(z, d, label=V)
        if prnt:
            print(
                f"Edge values for dimensionless luminosity distance for {V}-potential:"
            )
            print("z=, d_L=")
            print(z[-1], d[-1])
            print(z[0], d[0])
            print()

    plt.xlabel("$z$")
    plt.ylabel(r"$\frac{H_0}{c}d_L$")
    plt.title("Luminosity distance for quintessence models")
    plt.legend()
    plt.grid()
    plt.gca().invert_xaxis()  # invert x-axis
    plt.savefig(filename)


if __name__ == "__main__":
    # Plotting parameters
    N_i = np.log(1 / 3)  # characteristic initial time
    N_f = 0  # characteristic stop time

    plot_lumosity_distances(N_i, N_f)
