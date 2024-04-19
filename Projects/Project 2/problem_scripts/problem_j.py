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

from bbn import BBN, FIG_DIR, COLORS


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


def bayesian_probability(
    predicted: np.ndarray, observed: np.ndarray, error: np.ndarray
) -> float:
    """Calculates the Bayesian probability for an array with predicted values,
    compared to array with observed values and array with the observed errors.
    Equation (28) of the project.

    arguments:
        predicted: the predicted values
        observed: the observed values
        error: the error in the observed values

    returns:
        the Bayesian probability
    """
    return (
        1
        / (np.sqrt(2 * error ** (2 * len(predicted))))
        * np.exp(-xi_squared(predicted, observed, error))
    )


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
        _, Y = bbn.solve_BBN(T_i, T_f)

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
    y_min = (
        0.5e-10  # minimum value for y-axis in the middle Y_i/Y_p relic abundance plot
    )
    Y_min = 1e-20  # lower bound value for mass fractions Y_i
    Omega_b0_min = 1e-2  # minimum value for Omega_b0
    Omega_b0_max = 1  # maximum value for Omega_b0
    n = 4  # number of points for Omega_b0 before interpolation
    n_plot = 1000  # number of points for Omega_b0 after interpolation
    Omega_b0_vals = np.logspace(
        np.log10(Omega_b0_min), np.log10(Omega_b0_max), n
    )  # baryon density parameter today
    filename = os.path.join(FIG_DIR, "j_relic_abundances.png")  # plot filename
    figsize = (6, 5)  # figure size

    # Observed values for relic abundances Y_i/Y_p
    D_ab = 2.57e-5
    D_ab_err = 0.03e-5
    Li7_ab = 1.6e-10
    Li7_ab_err = 0.3e-10

    # Observed value for mass fraction 4Y_He4
    mass_frac_He4 = 0.254
    mass_frac_He4_err = 0.003

    # Interpolate
    Y_p_interp, Y_D_interp, Y_He4_interp, Y_Li7_interp = interpolate_Y_func_of_Omegab0(
        Omega_b0_vals, N_species=8, Y_min=Y_min
    )

    # Array to interpolate graph
    Omega_b0_arr = np.logspace(
        np.log10(Omega_b0_vals[0]), np.log10(Omega_b0_vals[-1]), n_plot
    )
    log_Omega_b0_arr = np.log10(Omega_b0_arr)

    # Plotting
    fig, axs = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        height_ratios=[1, 3, 1],
    )
    # Plot 4Y_He4
    axs[0].plot(
        Omega_b0_arr,
        4 * 10 ** (Y_He4_interp(log_Omega_b0_arr)),
        label="4Y_He4",
        color=COLORS[5],
    )

    # Plot errorbar area for observed value of 4Y_He4
    opacity = 0.3
    axs[0].fill_between(
        Omega_b0_arr,
        mass_frac_He4 - mass_frac_He4_err,
        mass_frac_He4 + mass_frac_He4_err,
        alpha=opacity,
        color=COLORS[5],
    )

    axs[0].set_ylabel(r"$4Y_{He4}$")
    axs[0].tick_params(axis="both", which="both", direction="in", top=True, right=True)
    axs[0].legend()
    axs[0].set_xscale("log")
    axs[0].grid(True)

    # Plot Y_i/Y_p for D, He3, and Li7
    axs[1].loglog(
        Omega_b0_arr, 10 ** (Y_D_interp(log_Omega_b0_arr)), label="D", color=COLORS[2]
    )
    axs[1].loglog(
        Omega_b0_arr,
        10 ** (Y_He4_interp(log_Omega_b0_arr)),
        label="He3",
        color=COLORS[4],
    )
    axs[1].loglog(
        Omega_b0_arr,
        10 ** (Y_Li7_interp(log_Omega_b0_arr)),
        label="Li7",
        color=COLORS[-2],
    )

    # Plot errorbar areas for observed values of Y_i/Y_p for D and Li7 (no error for He3)
    axs[1].fill_between(
        Omega_b0_arr,
        D_ab - D_ab_err,
        D_ab + D_ab_err,
        alpha=opacity,
        color=COLORS[2],
    )
    axs[1].fill_between(
        Omega_b0_arr,
        Li7_ab - Li7_ab_err,
        Li7_ab + Li7_ab_err,
        alpha=opacity,
        color=COLORS[-2],
    )

    axs[1].set_ylim(bottom=y_min)
    axs[1].set_ylabel(r"$Y_i/Y_p$")
    axs[1].tick_params(axis="both", which="both", direction="in", top=True, right=True)
    axs[1].legend()
    axs[1].grid(True)

    # Calculate Bayesian likelihood
    model = np.asarray(
        [
            10 ** (Y_D_interp(log_Omega_b0_arr)),
            10 ** (Y_He4_interp(log_Omega_b0_arr)),
            10 ** (Y_Li7_interp(log_Omega_b0_arr)),
        ]
    )
    observed = np.asarray([D_ab, mass_frac_He4, Li7_ab])
    error = np.asarray([D_ab_err, mass_frac_He4_err, Li7_ab_err])

    likelihood = bayesian_probability(model, observed, error)

    # Plot Bayesian likelihood
    axs[2].plot(Omega_b0_arr, likelihood)
    axs[2].set_xscale("log")
    axs[2].tick_params(axis="both", which="both", direction="in", top=True, right=True)
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_ylabel("Bayesian\nlikelihood")

    # Plot config
    fig.suptitle("Relic abundance analysis")
    fig.supxlabel(r"$\Omega_{b0}$")
    # plt.ylabel(r"$Y_i/Y_p$")
    # plt.tight_layout()
    # plt.subplots_adjust(hspace=0.1)
    plt.savefig(filename)
