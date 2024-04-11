import numpy as np

from constants import SI, CGS


class Background:
    """Class for the background functions, a(T) and H(T), and variable N_eff and Omega_r0"""

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