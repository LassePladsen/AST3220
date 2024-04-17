"""Script that solves problem j of the project

Plots the relative abundances compared to the proton/Hydrogen Y_i/Y_p 
for D, He3, and Li7, as functions of the baryon density parameter Omega_b0. 
Also finds the most likely Omega_b0 value. 
"""

import sys
import os
import time

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


if __name__ == "__main__":
    # Variables
    N_species = 8  # number of interacting atom species
    T_i = 1e11  # initial temperature [K]
    T_f = 1e7  # final temperature [K]
    ymin = 1e-11  # minimum value for y-axis
    n_points = 10  # number of points to use for Omega_b0_vals
    Omega_b0_vals = np.logspace(-2, 0, n_points)  # baryon density parameter today
    filename = os.path.join(FIG_DIR, "j_relic_abundances.png")  # plot filename

    # Observed values for relic abundances Y_i/Y_p
    D_ab_err = 2.57e-5
    D_ab_err = 0.03e-5
    Li7_ab = 1.6e-10
    Li7_ab_err = 0.3e-10 

    # Observed value for He4 mass fraction 4Y_He4
    mass_frac_He4 = 0.254
    mass_frac_He4_err = 0.003 

    # Time the execution time
    start = time.time()

    Y_model = []
    for Omega_b0 in Omega_b0_vals:
        # Initialize
        bbn = BBN(N_species, Omega_b0=Omega_b0)

        # Solve ode
        T, Y = bbn.solve_ode_system(T_i, T_f, n_points=n_points)

        print(Y.shape)
        # Extract values for final temperature
        Y = Y[-1]
        print(Y.shape)
        quit()

        # T and Be7 decays to respectively He3 and Li7
        Y[4] += Y[3]
        Y[6] += Y[7]

        # Set lower bound for mass fractions 


        # We only need p, D, He4, and Li7 from here on, to save computational time
        Y_p = Y[1]
        Y_D = Y[2]
        Y_He4 = Y[5]
        Y_Li7 = Y[6]
        Y = np.asarray([Y_p, Y_D, Y_He4, Y_Li7])


    # Interpolate in logspace
    # interp = interp1d(lnT, lnY, kind="cubic")

        Y_model.append(Y)
    
    # Interpolate in logspace
    Y_model = np.array(Y_model)
    interp = interp1d(np.log(Omega_b0_vals), np.log(Y_model), kind="cubic")

    # Plot


    plt.plot(Omega_b0, interp(Omega_b0), label="Interpolated")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()



    # Print timing
    print(f"\nExecution time: {time.time() - start:.2f} s")
