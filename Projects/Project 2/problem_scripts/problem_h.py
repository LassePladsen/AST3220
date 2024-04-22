"""Script that solves problem h of the project"""

import sys
from pathlib import Path

# Append path to bbn package
sys.path.append(str(Path(__file__).parents[1]))

from bbn.bbn import BBN, FIG_DIR

if __name__ == "__main__":
    # Variables
    N_species = 3  # number of interacting atom species
    T_i = 1e11  # initial temperature [K]
    T_f = 1e8  # final temperature [K]

    # Initialize
    bbn = BBN(N_species)

    # Solve ode
    bbn.solve_BBN(T_i, T_f)

    # Plot mass fractions
    filename = FIG_DIR / "h_mass_fractions.png"
    bbn.plot_mass_fractions(filename)
