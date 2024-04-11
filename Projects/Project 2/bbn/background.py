from abc import ABC

import numpy as np
from astropy import constants as ac

class Constants(ABC):
    """Abstract base class for constants"""
    T_0 = 2.725 # CMB temperature [K]
    H_0 = H_0 = 22.686e-19  # Hubble constant with h=0.7 [1/s]


class SI(Constants):
    """Class to hold constants used in the project given in SI units"""
    c = ac.c.value  # speed of light [m/s]
    k = ac.k_B.value  # Boltzmann constant [J/K]
    hbar = ac.hbar.value  # reduced Planck constant [J*s]
    G = ac.G.value  # gravitational constant [N*m^2/kg^2]
    m_p = ac.m_p.value  # proton mass [kg]
    m_n = ac.m_n.value  # neutron mass [kg]


class CGS(Constants):
    """Class to hold constants used in the project given in CGS units"""
    c = ac.c.value  # speed of light [m/s]
    k = ac.k_B.value  # Boltzmann constant [J/K]
    hbar = ac.hbar.value  # reduced Planck constant [J*s]
    G = ac.G.value  # gravitational constant [N*m^2/kg^2]
    m_p = ac.m_p.value  # proton mass [kg]
    m_n = ac.m_n.value  # neutron mass [kg]


class Background:
    """Class to store the background functions"""

    def __init__(self, unit: str = "CGS", N_eff: int = 3) -> None:
        """Initializes the class with the given parameters.

        arguments:
            unit: the unit system to use, "SI" or "CGS" (case insensitive)
            N_eff: effective number of neutrino species
        """
        if unit.lower() == "si":
            self.const = SI()
        elif unit.lower() == "cgs":
            self.const = CGS()
        else:
            raise ValueError("unit parameter must be either 'SI' or 'CGS'")

        self.N_eff = N_eff  # effective number of neutrino species
        self.Omega_r0 = (  # radiation density parameter today
            8
            * np.pi**3
            / 45
            * self.const.G
            / (self.const.H_0 * self.const.H_0)
            * (self.const.k * self.const.T_0) ** 4
            / (self.const.hbar**3 * self.const.c**5)
            * (1 + self.N_eff * 7 / 8 * (4 / 11) ** (4 / 3))
        )

    def a(self, T: float) -> float:
        """Scale factor as function of temperature

        arguments:
            T: temperature [K]

        returns:
            the scale factor
        """
        return self.const.T_0 / T

    def H(self, T: float) -> float:
        """Hubble parameter as function of temperature, equation (14) of the project

        arguments:
            T: temperature [K]

        returns:
            the Hubble parameter
        """
        return self.const.H_0 * np.sqrt(self.Omega_r0) * self.a(T) ** (-2)


"""if __name__ == '__main__':
    # Debug plot H(T)
    import matplotlib.pyplot as plt
    b = Background()
    T = np.logspace(11, 7)
    h = b.H(T)
    print(h[0])
    plt.loglog(T, h)
    plt.show()"""
