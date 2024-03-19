import os

from scipy.integrate import cumulative_trapezoid
import numpy as np
import matplotlib.pyplot as plt

from problem9 import FIGURES_DIR
from problem10 import hubble_parameter_lambdacdm
from problem13 import DATA_PATH, c, H_0, chi_squared


def lumonisity_integrand_lambdacdm(
    z: np.ndarray, Omega_m0: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Represents the integral over z of the dimensionless luminosity distance
    for the lambda-cdm model
    arguments:
        z: the redshift array
        Omega_m0: the matter density constant of today
    returns:
        The integrand values array
    """

    h = hubble_parameter_lambdacdm(z, Omega_m0)
    return 1 / h


def luminosity_distance_lambdacdm(
    z: np.ndarray, Omega_m0: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the dimensionless luminosity distance H_0d_L/c for the lambda-cdm model
    as a function of the redshift
    arguments:
        z: the redshift array
        Omega_m0: the matter density constant of today
    returns:
        The dimensionless luminosity distance
    """
    I = lumonisity_integrand_lambdacdm(z, Omega_m0)

    return cumulative_trapezoid(I, z, initial=0)


def plot_lambdacdm_chisquared(
    Omega_m0_vals: np.ndarray,
    filename: str = None,
    figsize: tuple[int, int] = (6, 4),
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
        filename = os.path.join(
            FIGURES_DIR,
            f"14_chi_squared_vals.png",
        )

    plt.figure(figsize=figsize)

    # Load data
    z, d_data, d_err = np.loadtxt(DATA_PATH, skiprows=5, unpack=True)  # [-, Gpc, Gpc]

    z_model = np.linspace(0, z[-1], 500)

    chi_vals = []
    for Omega_m0 in Omega_m0_vals:
        # Get model prediction
        d_model = (
            luminosity_distance_lambdacdm(z_model, Omega_m0) * c / H_0
        )  # Convert to Gpc

        # Cut model to data range
        # For each z from data, find corresponding d_model value
        tol = 1e-3
        d = []
        for zi in z:
            ind = np.where(abs(zi - z_model) < tol)[0][0]
            d.append(d_model[ind])

        # Compare to data, append to list
        chi_vals.append(chi_squared(d, d_data, d_err))

    # Plot
    plt.plot(Omega_m0_vals, chi_vals)
    plt.xlabel(r"$\Omega_{m0}$")
    plt.ylabel("$\chi^2$")
    plt.title("$\chi^2$-values for the $\Lambda$-CDM model")
    plt.grid()
    plt.savefig(filename)

    if prnt:
        print(f"Lowest chi-squared value: {np.min(chi_vals):.0f}")
        print(f"Corresponding Omega_m0 value: {Omega_m0_vals[np.argmin(chi_vals)]:.3f}")


if __name__ == "__main__":
    Omega_m0_vals = np.linspace(0, 1, 100)
    plot_lambdacdm_chisquared(Omega_m0_vals)
