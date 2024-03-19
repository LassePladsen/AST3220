import os

import numpy as np
import matplotlib.pyplot as plt

from problem9 import N_i, N_f, FIGURES_DIR
from problem12 import luminosity_distance_quintessence

# Path to the data file
DATA_PATH = os.path.join(os.path.dirname(__file__), "sndata.txt")

# Constants
H_0 = 70e6  # Hubble constant [m/(s Gpc)]
c = 3e8  # speed of light [m/s]


def chi_squared(prediction: np.ndarray, data: np.ndarray, err: np.ndarray) -> float:
    """
    Calculate the chi-squared value for a model prediction given the data and the errors

    arguments:
        prediction: the model prediction [Gpc]
        data: the observed data [Gpc]
        err: the errors in the observed data [Gpc]

    returns:
        The chi-squared value
    """
    return np.sum(((prediction - data) / err) ** 2)


def plot_luminosity_distances(
    filename: str = None,
    figsize: tuple[int, int] = (6, 4),
) -> None:
    """Plots the luminosity distance using the data and the quintessence models
    as a function of the redshift, in Gpc units

    arguments:
        filename: the filename to save the plot figure
        figsize: the plot figure size

    returns:
        none
    """

    # Default filename
    if not filename:
        filename = os.path.join(
            FIGURES_DIR,
            f"13_luminosity_distances.png",
        )

    plt.figure(figsize=figsize)

    # Load data
    z, d_data, d_err = np.loadtxt(DATA_PATH, skiprows=5, unpack=True)  # [-, Gpc, Gpc]

    # Plot the two quintessence models
    for V in ["power", "exponential"]:
        # d_model = luminosity_distance_quintessence(V, 2, z=z)[-1]
        z_model, d_model = luminosity_distance_quintessence(V, z[-1], N_i, N_f)
        plt.plot(z_model, d_model * c / H_0, label=V)  # convert to Gpc when plotting

    # Plot the data
    plt.errorbar(z, d_data, yerr=d_err, fmt=".", label="Data points", color="gray")

    plt.xlabel("$z$")
    plt.ylabel("$d_L$ [Gpc]")
    plt.title("Luminosity distance comparision for quintessence models")
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend()
    plt.grid()
    plt.savefig(filename)


def print_chi_squared_values() -> None:
    """Prints the chi-squared values for the two quintessence models

    arguments:
        none

    returns:
        none
    """

    # Load data
    z, d_data, d_err = np.loadtxt(DATA_PATH, skiprows=5, unpack=True)  # [-, Gpc, Gpc]

    # Model predictions
    for V in ["power", "exponential"]:
        z_model, d_model = luminosity_distance_quintessence(V, z[-1], N_i, N_f)
        d_model *= c / H_0  # convert to Gpc

        # Cut model to data range
        # For each z from data, find corresponding d_model value
        tol = 1e-5
        d = []
        for zi in z:
            ind = np.where(abs(zi - z_model) < tol)[0][0]
            d.append(d_model[ind])

        print(
            f"Chi-squared value for {V}-potential: {chi_squared(d, d_data, d_err):.0f}"
        )


if __name__ == "__main__":
    plot_luminosity_distances()
    print_chi_squared_values()
