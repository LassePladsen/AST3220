import numpy as np
from scipy.integrate import quad

from expressions import eos_integral_N
from problem9 import density_parameters
from problem10 import hubble_parameter_quintessence, hubble_parameter_lambdacdm

V = "power"
z, Omega_m, Omega_r, Omega_phi = density_parameters(V)
Omega_m0 = Omega_m[-1]
Omega_r0 = Omega_r[-1]
Omega_phi0 = Omega_phi[-1]
# _, h = hubble_parameter_quintessence("power")
# h = hubble_parameter_lambdacdm(z)

def hubble(zi):
    Ni = np.log(1 / (1 + zi))
    np.sqrt(
        Omega_m0 * (1 + zi) ** 3
        + Omega_r0 * (1 + zi) ** 4
        + Omega_phi0 * np.exp(quad(eos_integral_N, Ni, 0)[0])
    )

def I(zi):
    return 1/( (1+zi)*hubble(zi) )

# print(I(z[5]))
print(quad(I, 0, np.inf))
