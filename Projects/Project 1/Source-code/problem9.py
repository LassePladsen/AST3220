from expressions import xi, ode_system

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def solve_ode_system(V: str):
    """Solves the ode system

    arguments:
        V: the potential function in ["power", "exponential"]

    returns:
        # TODO
    """

    # Initial conditions and variables
    N_i = np.log(1 / (1 + 2e7))  # characteristic initial time
    N_f = 0  # characteristic stop time
    if V.lower() == "power":
        x1_i = 5e-5
        x2_i = 1e-8
        x3_i = 0.9999
        lmbda_i = 1e9
    elif V.lower() == "exponential":
        x1_i = 0
        x2_i = 5e-13
        x3_i = 0.9999
        lmbda_i = xi
    else:
        raise ValueError("V-string value not recognized")

    sol = solve_ivp(
        ode_system,
        [N_i, N_f],
        [x1_i, x2_i, x3_i, lmbda_i],
        args=(V,),
        # method="DOP853",
        # rtol=1e-8,
        # atol=1e-8,
        # first_step=abs(N_i)/100,
    )
    print(sol)


solve_ode_system("power")

# a = np.linspace(np.log(1 / (1 + 2e7)), 0, 100)
# print(a[4]-a[5])
# print(np.log(1 / (1 + 2e7))/99)
