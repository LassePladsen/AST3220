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

from bbn import BBN, FIG_DIR

if __name__ == "__main__":
    # Variables
    T_i = 1e11  # initial temperature [K]
    T_f = 1e7  # final temperature [K]
    n = 4  # number of points for Omega_b0 before interpolation TODO: RAISE
    Omega_b0_min = 1e-2  # minimum value for Omega_b0
    Omega_b0_max = 1  # maximum value for Omega_b0
    filename = os.path.join(FIG_DIR, "j_relic_abundances.png")

    Omega_b0_vals = np.logspace(np.log10(Omega_b0_min), np.log10(Omega_b0_max), n)

    # Plot relic abundances
    BBN.plot_relic_abundances(T_i, T_f, Omega_b0_vals, filename=filename)
