from functools import lru_cache

import numpy as np


class SI:
    """Class to hold constants used in the project given in SI units"""

    def __init__(self) -> None:
        """Initializes the class with the constants"""
        self.c = 2.9979e8  # speed of light [m/s]
        self.k = 1.380649e-23  # Boltzmann constant [J/K]
        self.hbar = 6.62607015e-34  # reduced Planck constant [J*s]
        self.H_0 = 22.686e-19  # Hubble constant [1/s]
        self.T_0 = 2.725  # CMB temperature [K]
        self.G = 6.67430e-11  # gravitational constant [N*m^2/kg^2]
        self.m_p = 1.67262192369e-27  # proton mass [kg]
        self.m_n = 1.67492749804e-27  # neutron mass [kg]


class CGS:
    """Class to hold constants used in the project given in CGS units"""

    def __init__(self, N_eff: int = 3) -> None:
        """Initializes the class with the constants

        arguments:
            N_eff: effective number of neutrino species
        """
        self.c = 2.9979e10  # speed of light [cm/s]
        self.k = 1.380649e-23  # Boltzmann constant [J/K]
        self.hbar = 6.62607015e-34  # reduced Planck constant [J*s]
        self.H_0 = 22.686e-12  # Hubble constant [1/s]
        self.T_0 = 2.725  # CMB temperature [K]
        self.G = 6.67430e-8  # gravitational constant [cm^3/g/s^2]
        self.m_p = 1.67262192369e-24  # proton mass [g]
        self.m_n = 1.67492749804e-24  # neutron mass [g]
        self.N_eff = N_eff  # effective number of neutrino species
        self.Omega_r0 = (  # radiation density parameter today
            8
            * np.pi**3
            / 45
            * self.G
            / (self.H_0 * self.H_0)
            * (self.k * self.const.T_0) ** 4
            / (self.hbar**3 * self.c**5)
            * (1 + self.N_eff * 7 / 8 * (4 / 11) ** (4 / 3))
        )


class Background:
    """Class to store the background functions"""

    def __init__(self, unit: str = "SI", N_eff: int = 3) -> None:
        """Initializes the class with the given parameters.

        arguments:
            unit: the unit system to use, "SI" or "CGS" (case insensitive)
            N_eff: effective number of neutrino species
        """
        if unit.lower() == "si":
            self.const = SI(N_eff)
        elif unit.lower() == "cgs":
            self.const = CGS(N_eff)
        else:
            raise ValueError("unit parameter must be either 'SI' or 'CGS'")

    def Y_n_equi(self, T: float) -> float:
        """Thermal equilibrium value of relative number density of neutrons Y_n, equation (16) of the project.

        arguments:
            T: temperature [K]

        returns:
            the thermal equilibrium value of Y_n
        """
        return 1 / (
            1
            + np.exp(
                (self.const.m_n - self.const.m_p)
                * self.const.c
                * self.const.c
                / (self.const.k * T)
            )
        )

    def Y_p_equi(self, T: float) -> float:
        """Thermal equilibrium value of relative number density of protons Y_p, equation (17) of the project.

        arguments:
            T: temperature [K]

        returns:
            the thermal equilibrium value of Y_n
        """
        return 1 - self.Y_n_equi(T)

    @lru_cache
    def a(self, T: float) -> float:
        """Scale factor as function of temperature

        arguments:
            T: temperature [K]

        returns:
            the scale factor
        """
        return self.const.T_0 / T

    @lru_cache
    def H(self, T: float) -> float:
        """Hubble parameter as function of temperature, equation (14) of the project

        arguments:
            T: temperature [K]

        returns:
            the Hubble parameter
        """
        return self.const.H_0 * np.sqrt(self.const.Omega_r0) * self.a(T) ** (-2)
