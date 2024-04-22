"""Script that solves problem u of the project"""

import sys
from pathlib import Path

# Append path to bbn package
sys.path.append(str(Path(__file__).parents[1]))

from bbn import FIG_DIR
from bbn.bbn import BBN

if __name__ == "__main__":
    # Variables
    N_species = 8  # number of interacting atom species
    T_i = 1e11  # initial temperature [K]
    T_f = 1e7  # final temperature [K]
    ymin = 1e-11  # minimum value for y-axis

    # Initialize
    bbn = BBN(N_species)

    # Solve ode
    bbn.solve_BBN(T_i, T_f)

    # Plot mass fractions
    filename = FIG_DIR / "i_mass_fractions.png"
    bbn.plot_mass_fractions(filename, ymin=ymin, plot_equilibrium=False)
