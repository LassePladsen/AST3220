import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumulative_trapezoid

from problem9 import density_parameters, eos_parameter
from expressions import eos_integral_N

# Parameters
z_i = 2e7  # initia redshift 
z_f = 0    # final redshift
n = 100    # number of points
z = np.linspace(z_i, z_f, n)


def hubble_parameter(z: float, V: str) -> float:
    """Returns the characteristic Hubble parameter H/H_0 as a function of the redshift z (eq. 7)

    arguments:
        z: the redshift
        V: the potential function in ["power", "exponential"]

    returns:
        the value of H/H_0 for the given redshift z
    """
    # Density parameters
    _, Omega_m, Omega_r, Omega_phi = density_parameters(V)
    Omega_m0 = Omega_m[-1]
    Omega_r0 = Omega_r[-1]
    Omega_phi0 = Omega_phi[-1]

    # Convert redshift z to characteristic time N
    N = np.log(1 / (1 + z))

    return np.sqrt(
        Omega_m0 * (1 + z) ** 3
        + Omega_r0 * (1 + z) ** 4 + Omega_phi0 * np.exp(quad(eos_integral_N, N, 0)[0])
    )


def plot_hubble_parameter(V: str, filename=None, figsize=True, prnt=True) -> None:
    """Plots the characteristic Hubble parameter H/H_0 as a function of the redshift z

    arguments:
        V: the potential function in ["power", "exponential"]
        filename: the name of the file to save the plot
        figsize: the size of the figure
        prnt: if True, print the value of H/H_0 at z=0
    
    returns:
        None
    """

    if not filename:
        filename = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "Figures",
                f"10_hubble_parameter_{V}.png",
            )
        )

    y = []
    for zi in z:
        y.append(hubble_parameter(zi, V))

    print(z[-1], y[-1])

    plt.figure()
    plt.title(f"Hubble parameter for {V}-potential")
    plt.plot(z, y)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("z")
    plt.ylabel("$H/H_0$")
    plt.grid()
    plt.show()

V = "power"

# z, omega_phi = eos_parameter(V)
# N = np.log(1/(1+z))
# print(N)
# print(quad(eos_integral_N, N[], 0)[0])
# print(np.shape(N), np.shape(eos_integral_N(omega_phi)))
# print(cumulative_trapezoid(N, eos_integral_N(omega_phi)))
plot_hubble_parameter(V)
