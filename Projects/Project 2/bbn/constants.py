"""Module for the constants used in the project"""

from abc import ABC

import numpy as np
from astropy import constants as ac


class Constants(ABC):
    """Abstract base class for constants"""

    T_0 = 2.725  # CMB temperature [K]
    H_0 = H_0 = 22.686e-19  # Hubble constant with h=0.7 [1/s]

    def __init__(self) -> None:
        # Calculate critical density today
        self.rho_c0 = 3 * self.H_0 * self.H_0 / (8 * np.pi * self.G)


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
