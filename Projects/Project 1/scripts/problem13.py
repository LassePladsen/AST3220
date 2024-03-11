import os

import numpy as np
import matplotlib.pyplot as plt

from problem12 import luminosity_distance_quintessence

# Path to the data file
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "luminosity_distances.txt"
)

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
        filename = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "Figures",
                f"13_luminosity_distances.png",
            )
        )

    plt.figure(figsize=figsize)

    # Load data
    z, d_data, d_err = np.loadtxt(DATA_PATH, skiprows=5, unpack=True)  # [-, Gpc, Gpc]

    # Plot the two quintessence models
    for V in ["power", "exponential"]:
        d_model = luminosity_distance_quintessence(V, z=z)[-1]
        plt.plot(z, d_model * c / H_0, label=V)  # convert to Gpc when plotting

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
        d_model = luminosity_distance_quintessence(V, z=z)[-1]
        d_model *= c / H_0  # convert to Gpc
        print(
            f"Chi-squared value for {V}-potential: {chi_squared(d_model, d_data, d_err)}"
        )


if __name__ == "__main__":
    plot_luminosity_distances()
    print_chi_squared_values()
