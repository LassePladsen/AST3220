import os

import numpy as np

from problem12 import luminosity_distance_quintessence

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "luminosity_distances.txt"
)

# Constants
H_0 = 70  # Hubble constant [m/s/Gpc]
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


def print_chi_squared_values() -> None:
    """Prints the chi-squared values for the two quintessence models

    arguments:
        none

    returns:
        none
    """

    # Load data
    z, d_L, err = np.loadtxt(DATA_PATH, skiprows=5, unpack=True)  # [-, Gpc, Gpc]

    # Characteristic time interval
    N_i = np.log(1 / (z[-1] + 1))
    N_f = np.log(1 / (z[0] + 1))

    # Model predictions
    for V in ["power", "exponential"]:
        d_L_model = luminosity_distance_quintessence(V, N_i, N_f, n_points=len(z))[-1]
        d_L_model *= c / H_0  # convert to Gpc
        print(d_L_model)
        print(d_L)
        print(
            f"Chi-squared value for {V}-potential: {chi_squared(d_L_model, d_L, err)}"
        )


if __name__ == "__main__":
    print_chi_squared_values()
