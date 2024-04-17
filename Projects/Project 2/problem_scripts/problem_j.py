"""Script that solves problem j of the project

Plots the relative abundances compared to the proton/Hydrogen Y_i/Y_p 
for D, He3, and Li7, as functions of the baryon density parameter Omega_b0. 
Also finds the most likely Omega_b0 value. 
"""

import sys
import os

# Append path to bbn package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bbn")))

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from bbn import BBN, FIG_DIR


def xi_squared(predicted: np.ndarray, observed: np.ndarray, error: np.ndarray) -> float:
    """Calculates the chi-squared value between the predicted and observed values.

    arguments:
        predicted: the predicted values
        observed: the observed values
        error: the error in the observed values

    returns:
        the chi-squared value
    """
    return np.sum(((predicted - observed) / error) ** 2)


def interpolate_Y_func_of_Omegab0(
    Omega_b0_vals: np.ndarray, N_species: int = 8, Y_min: float = 1e-20
) -> tuple[callable, callable, callable, callable]:
    """Interpolates the mass fractions Y_i as a function of Omega_b0 to give a smooth
    function graph. Interpolates in logspace (base 10)

    arguments:
        Omega_b0_vals: array of Omega_b0 values to interpolate
        N_species: number of interacting atom species
        Y_min: lower bound value for mass fractions, everything below this value is set to Y_min

    returns:
        Tuple containing the logspace interpolated mass fractions functions logY_i(logOmega_b0)
        for i = D, He3, He4, and Li7
    """
    Y_model = np.zeros((4, len(Omega_b0_vals)))
    for i, Omega_b0 in enumerate(Omega_b0_vals):
        # Initialize
        bbn = BBN(N_species, Omega_b0=Omega_b0)

        # Solve ode
        _, Y = bbn.solve_ode_system(T_i, T_f)

        # Extract mass fraction values for final temperature
        Y = Y[:, -1]

        # T and Be7 decays to respectively He3 and Li7
        Y[4] += Y[3]
        Y[6] += Y[7]

        # Set lower bound for mass fractions
        Y[Y < Y_min] = Y_min

        # Extract mass fractions
        Y_p = Y[1]
        Y_D = Y[2]
        Y_He4 = Y[5]
        Y_Li7 = Y[6]
        Y = np.asarray([Y_p, Y_D, Y_He4, Y_Li7])

        # Append processed mass fractions
        Y_model[:, i] = Y

    # Interpolate each function in logspace
    kind = "cubic"
    Y_p, Y_D, Y_He4, Y_Li7 = Y_model

    Y_p_interp = interp1d(np.log10(Omega_b0_vals), np.log10(Y_p), kind=kind)
    Y_D_interp = interp1d(np.log10(Omega_b0_vals), np.log10(Y_D), kind=kind)
    Y_He4_interp = interp1d(np.log10(Omega_b0_vals), np.log10(Y_He4), kind=kind)
    Y_Li7_interp = interp1d(np.log10(Omega_b0_vals), np.log10(Y_Li7), kind=kind)

    return Y_p_interp, Y_D_interp, Y_He4_interp, Y_Li7_interp


if __name__ == "__main__":
    # Variables
    T_i = 1e11  # initial temperature [K]
    T_f = 1e7  # final temperature [K]
    y_min = 1e-11  # minimum value for y-axis
    Y_min = 1e-20  # lower bound value for mass fractions Y_i
    Omega_b0_min = 1e-2  # minimum value for Omega_b0
    Omega_b0_max = 1  # maximum value for Omega_b0
    n = 10  # number of points for Omega_b0 before interpolation
    n_plot = 1000  # number of points for Omega_b0 after interpolation
    Omega_b0_vals = np.logspace(
        np.log10(Omega_b0_min), np.log10(Omega_b0_max), n
    )  # baryon density parameter today
    filename = os.path.join(FIG_DIR, "j_relic_abundances.png")  # plot filename

    # Observed values for relic abundances Y_i/Y_p
    D_ab_err = 2.57e-5
    D_ab_err = 0.03e-5
    Li7_ab = 1.6e-10
    Li7_ab_err = 0.3e-10

    # Observed value for He4 mass fraction 4Y_He4
    mass_frac_He4 = 0.254
    mass_frac_He4_err = 0.003

    # Interpolate
    Y_p_interp, Y_D_interp, Y_He4_interp, Y_Li7_interp = interpolate_Y_func_of_Omegab0(
        Omega_b0_vals, N_species=8, Y_min=Y_min
    )

    # Array to interpolate graph
    Omega_b0_arr = np.logspace(np.log10(Omega_b0_vals[0]), np.log10(Omega_b0_vals[-1]), n_plot)
    log_Omega_b0_arr = np.log10(Omega_b0_arr)

    # Plot Y_i/Y_p for D, He3, and Li7
    plt.loglog(Omega_b0_arr, 10**(Y_D_interp(log_Omega_b0_arr)), label="D")
    plt.loglog(Omega_b0_arr, 10**(Y_He4_interp(log_Omega_b0_arr)), label="He3")
    plt.loglog(Omega_b0_arr, 10**(Y_Li7_interp(log_Omega_b0_arr)), label="Li7")

    # Plot config
    plt.legend()
    plt.xlabel(r"$\Omega_{b0}$")
    plt.ylabel("r$Y_i$")
    # plt.ylabel(r"$Y_i/Y_p$")
    plt.grid()
    plt.savefig(filename)
