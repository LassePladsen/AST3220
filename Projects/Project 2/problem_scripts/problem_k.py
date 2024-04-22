"""Script that solves problem k of the project

Plots the relative abundances compared to the proton/Hydrogen Y_i/Y_p 
for D, He3, and Li7, as functions of the number of neutrinos species N_eff. 
Also finds the most likely N_eff value. 
"""

import sys
from pathlib import Path

# Append path to bbn package
sys.path.append(str(Path(__file__).parents[1]))

import numpy as np

from bbn import FIG_DIR
from bbn.bbn import BBN

if __name__ == "__main__":
    # Variables
    T_i = 1e11  # initial temperature [K]
    T_f = 1e7  # final temperature [K]
    N_eff_vals = np.linspace(1, 5, 5)  # N_eff values to test
    filename = FIG_DIR / "k_relic_abundances.png"

    # Plot relic abundances
    BBN.plot_relic_abundances_Neff(T_i, T_f, N_eff_vals, filename=filename)
