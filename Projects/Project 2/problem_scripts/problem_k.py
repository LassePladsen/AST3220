"""Script that solves problem k of the project

Plots the relative abundances compared to the proton/Hydrogen Y_i/Y_p 
for D, He3, and Li7, as functions of the number of neutrinos species N_eff. 
Also finds the most likely N_eff value. 
"""

import sys
import os

# Append path to bbn package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bbn")))

import numpy as np

from bbn import BBN, FIG_DIR

if __name__ == "__main__":
    # Variables
    T_i = 1e11  # initial temperature [K]
    T_f = 1e7  # final temperature [K]
    n = 5  # number of points for Omega_b0 before interpolation TODO: RAISE
    N_eff_min = 1  # minimum value for Omega_b0
    N_eff_max = 5  # maximum value for Omega_b0
    filename = os.path.join(FIG_DIR, "k_relic_abundances.png")

    N_eff_vals = np.linspace(N_eff_min, N_eff_max, n)

    # Plot relic abundances
    BBN.plot_relic_abundances_Neff(T_i, T_f, N_eff_vals, filename=filename)
