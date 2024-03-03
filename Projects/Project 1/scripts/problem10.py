import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from problem9 import density_parameters, N_i, N_f


def eos_integrand(omega_phi: float) -> float:
    """Represents the integrand over N of the equation of state (the projects equation 7)

    arguments:
        omega_phi: the eos parameter

    returns:
        the value at given input
    """
    return 3 * (1 + omega_phi)


def hubble_parameter_quintessence(
    V: str, N_i: float, N_f: float, n_points: int = int(1e6)
) -> tuple[np.ndarray, np.ndarray]:
    """Characteristic Hubble parameter H/H_0 as a function of the redshift z (eq. 7)
    for the quintessence model, using a given potential function name

    arguments:
        V: the potential function in ["power", "exponential"]
        N_i: the characteristic initial time
        N_f: the characteristic stop time
        n_points: the number of points to evaluate the integral

    returns:
        z: the redshift z
        H: the values of H/H_0 for each z from the z-interval
    """
    # Density parameters
    z, Omega_m, Omega_r, Omega_phi = density_parameters(V, N_i, N_f, n_points)
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
            + Omega_phi0 * np.exp(quad(eos_integrand, N[i], 0)[0])
        )

    return z, H


def hubble_parameter_lambdacdm(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Characteristic Hubble parameter H/H_0 as a function of the redshift z (eq. 7)
    for the spatially flat (k=0) lambda-cdm model with Omega_m0=0.3,
    omega_lambda0 ~ 0.7, and Omega_r0=Omega_k0=0

    arguments:
        z: the redshift z array

    returns:
        H: the values of H/H_0 for each z from the z-interval
    """
    # Density parameters
    Omega_m0 = 0.3
    Omega_lambda0 = 0.7

    # Characteristic Hubble parameter
    h = np.sqrt(Omega_m0 * (1 + z) ** 3 + Omega_lambda0)

    return h


def plot_hubble_parameters(
    filename: str = None, figsize: tuple[int, int] = (6, 5), prnt=True
) -> None:
    """Plots the characteristic Hubble parameter H/H_0 as a function of the redshift z,
    for both the power-law and the exponential potentials in addition to the lambda-CDM model

    arguments:
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
                f"10_hubble_parameters.png",
            )
        )

    plt.figure(figsize=figsize)
    plt.title(f"Hubble parameter for quintessence and Lambda-CDM")

    # Quintessence models
    for V in ["power", "exponential"]:
        z, h = hubble_parameter_quintessence(V, N_i, N_f)
        plt.plot(z, h, label=f"{V}-potential")
        if prnt:
            print(f"Edge values for {V}-potential:")
            print("z=, H/H_0=")
            print(z[-1], h[-1])
            print(z[0], h[0])
            print()

    # Lambda-CDM model
    h = hubble_parameter_lambdacdm(z)
    plt.plot(z, h, label=r"$\Lambda$-CDM")
    if prnt:
        print(f"Edge values for lambda-CDM:")
        print("z=, H/H_0=")
        print(z[-1], h[-1])
        print(z[0], h[0])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("z")
    plt.ylabel("$H/H_0$")
    plt.grid()
    plt.legend()
    plt.savefig(filename)


if __name__ == "__main__":
    plot_hubble_parameters()
