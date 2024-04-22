"""Script that solves problem f of the project"""

import sys
from pathlib import Path

# Append path to bbn package
sys.path.append(str(Path(__file__).parents[1]))

from bbn import FIG_DIR
from bbn.bbn import BBN

if __name__ == "__main__":
    # Variables
    N_species = 2  # number of interacting atom species
    T_i = 1e11  # initial temperature [K]
    T_f = 1e8  # final temperature [K]

    # Initialize
    BBN = BBN(N_species)

    # Solve ode
    BBN.solve_BBN(T_i, T_f)

    # Plot mass fractions
    filename = FIG_DIR / "f_mass_fractions.png"
    BBN.plot_mass_fractions(filename)
