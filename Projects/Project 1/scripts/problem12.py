import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

from problem10 import hubble_parameter_quintessence


def lumonisity_integrand_quintessence(
    V: str,
    N_i: float = None,
    N_f: float = None,
    n_points: int = int(1e6),
    z: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Represents the integral over z of the dimensionless luminosity distance (eq. S)
    for the quintessence model

    arguments:
        V: the quintessence potential function in ["power", "exponential"]
        N_i: the characteristic initial time
        N_f: the characteristic stop time
        n_points: the number of points to evaluate the integral
        z: optionally give the redshift array instead of start and stop time

    returns:
        z: The redshift array
        I: The integrand values array
    """

    if z is None:
        z, H = hubble_parameter_quintessence(V, N_i, N_f, n_points)
    else:
        z, H = hubble_parameter_quintessence(V, z=z)

    z = np.flip(z)
    return z, 1 / H


def luminosity_distance_quintessence(
    V: str,
    z_i: float = None,
    z_f: float = None,
    n_points: int = int(1e6),
    z: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the dimensionless luminosity distance H_0 d_L/c for the quintessence model
    as a function of the redshift z

    arguments:
        V: the quintessence potential function in ["power", "exponential"]
        z_i: the initial redshift
        z_f: the final redshift
        n_points: the number of points to evaluate the integral
        z: optionally give the redshift array instead of start and stop time

    returns:
        z: The redshift array
        d: The dimensionless luminosity distances array
    """
    if z is None:
        N_i = np.log(1 / (1 + z_f))
        N_f = np.log(1 / (1 + z_i))
        z, I = lumonisity_integrand_quintessence(V, N_i, N_f, n_points)
    else:
        z, I = lumonisity_integrand_quintessence(V, z=z)

    return z, (1 + z) * cumulative_trapezoid(I, z, initial=0)


def plot_lumosity_distances(
    z_i: float,
    z_f: float,
    filename: str = None,
    figsize: tuple[int, int] = (6, 4),
    prnt: bool = True,
) -> None:
    """Plots the dimensionless luminosity distance for the two quintessence models

    arguments:
        z_i: the initial redshift
        z_f: the final redshift
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
        z, d = luminosity_distance_quintessence(V, z_i, z_f)
        plt.plot(z, d, label=V)
        if prnt:
            print(
                f"Edge values for dimensionless luminosity distance for {V}-potential:"
            )
            print("z=, H0/c d_L=")
            print(f"{z[0]:3g}, {d[0]:6g}")
            print(f"{z[-1]:3g}, {d[-1]:6.4f}")
            print()

    plt.xlabel("$z$")
    plt.ylabel(r"$\frac{H_0}{c}d_L$")
    plt.title("Luminosity distance for quintessence models")
    plt.legend()
    plt.grid()
    plt.gca().invert_xaxis()
    plt.savefig(filename)


if __name__ == "__main__":
    # Plotting parameters
    z_i = 0  # initial redshift
    z_f = 2  # final redshift

    plot_lumosity_distances(z_i, z_f)
