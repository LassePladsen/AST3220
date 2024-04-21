"""Script that solves problem d of the project"""

import sys
from pathlib import Path

import numpy as np

# Append path to bbn package
sys.path.append(str(Path(__file__).parents[1]))

from bbn import constants

if __name__ == "__main__":
    u = constants.SI()
    H0sqrtOmega_r0 = np.sqrt(  # H_0 * sqrt(Omega_r0)
        8
        * np.pi**3
        / 45
        * u.G
        * (u.k * u.T_0) ** 4
        / (u.hbar**3 * u.c**5)
        * (1 + 3 * 7 / 8 * (4 / 11) ** (4 / 3))
    )
    print(f"H_0 * sqrt(Omega_r0): {H0sqrtOmega_r0:.4e}")
    for T in [1e10, 1e9, 1e8]:
        t = 1 / (2 * H0sqrtOmega_r0) * (u.T_0 / T) ** 2
        print(f"Time at T = {T:.4e}: {t:.4e} s")
