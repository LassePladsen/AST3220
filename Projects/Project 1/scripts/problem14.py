import os

from scipy.integrate import cumulative_trapezoid
import numpy as np
import matplotlib.pyplot as plt

from problem10 import hubble_parameter_lambdacdm
from problem13 import DATA_PATH, c, H_0, chi_squared


def lumonisity_integrand_lambdacdm(
    z, Omega_m0: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Represents the integral over z of the dimensionless luminosity distance (eq. S)
    for the lambda-cdm model
    arguments:
        z: the redshift array
        Omega_m0: the Omega_m0 parameter
    returns:
        The integrand values array
    """

    h = hubble_parameter_lambdacdm(np.flip(z), Omega_m0)
    return 1 / h


def luminosity_distance_lambdacdm(
    z, Omega_m0: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the dimensionless luminosity distance H_0d_L/c for the lambda-cdm model
    as a function of the redshift
    arguments:
        z: the redshift array
        Omega_m0: the Omega_m0 parameter
    returns:
        The dimensionless luminosity distance
    """
    I = lumonisity_integrand_lambdacdm(z, Omega_m0)

    return cumulative_trapezoid(I, z, initial=0)


def plot_lambdacdm_chisquared(
    Omega_m0_vals: np.ndarray,
    filename: str = None,
    figsize: tuple[int, int] = (7, 4),
    prnt: bool = True,
) -> None:
    """Plots the chi-squared value as a function of the Omega_m0 parameter for
    the lambda-cdm model, comparing data and the model predictions

    arguments:
        Omega_m0_arr: the array of Omega_m0 values to plot for
        filename: the filename to save the plot figure
        figsize: the plot figure size'
        prnt: if true, prints the lowest value of chi-squared and the corresponding Omega_m0 value
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
                f"14_chi_squared_vals.png",
            )
        )

    plt.figure(figsize=figsize)

    # Load data
    z, d_data, d_err = np.loadtxt(DATA_PATH, skiprows=5, unpack=True)  # [-, Gpc, Gpc]

    chi_vals = []
    for Omega_m0 in Omega_m0_vals:
        # Get model prediction
        d = luminosity_distance_lambdacdm(z, Omega_m0) * c / H_0  # Convert to Gpc

        # Compare to data, append to list
        chi_vals.append(chi_squared(d, d_data, d_err))

    # Plot
    plt.plot(Omega_m0_vals, chi_vals)
    plt.xlabel(r"$\Omega_{m0}$")
    plt.ylabel("$\chi^2$")
    plt.title("$\chi^2$-values for the $\Lambda$-CDM model")
    plt.grid()
    plt.gca().invert_xaxis()  # invert x-axis
    plt.savefig(filename)

    if prnt:
        print(f"Lowest chi-squared value: {min(chi_vals)}")
        print(f"Corresponding Omega_m0 value: {Omega_m0_vals[np.argmin(chi_vals)]}")


if __name__ == "__main__":
    Omega_m0_vals = np.linspace(0, 1, 100)
    plot_lambdacdm_chisquared(Omega_m0_vals)
