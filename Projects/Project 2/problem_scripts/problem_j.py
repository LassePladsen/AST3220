"""Script that solves problem j of the project

Plots the relative abundances compared to the proton/Hydrogen Y_i/Y_p 
for D, He3, and Li7, as functions of the baryon density parameter Omega_b0. 
Also finds the most likely Omega_b0 value. 
"""

from pathlib import Path
import sys

import numpy as np

# Append path to bbn package
sys.path.append(str(Path(__file__).parents[1]))

from bbn.bbn import BBN, FIG_DIR

if __name__ == "__main__":
    # Variables
    T_i = 1e11  # initial temperature [K]
    T_f = 1e7  # final temperature [K]
    Omega_b0_vals = np.logspace(-2, 0, 20)  # Omega_b0 values to test
    filename = FIG_DIR / "j_relic_abundances.png"

    # Plot relic abundances
    BBN.plot_relic_abundances_Omegab0(T_i, T_f, Omega_b0_vals, filename=filename)
