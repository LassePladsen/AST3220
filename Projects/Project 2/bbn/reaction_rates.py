"""Reaction rates for the early universe."""

from functools import lru_cache

from scipy.integrate import quad
import numpy as np


class ReactionRates:
    """Class describing reaction rates between the particle species in the early universe."""

    def __init__(self, tau: float = 1700) -> None:
        """Initializes the class with a given neutron lifetime.
        6
                arguments:
                    tau: free neutron decay time [s]
        """

        self.tau = tau  # free neutron decay time [s]
        self.q = 2.53  # mass difference ratio [(m_n - m_p) / m_e]

    @lru_cache
    def _lambda_np(self, T_9: float, q_sign: int = 1) -> float:
        """Describes the reaction rate of n -> p, equation (12) of the project,
        described in a.3 table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            q: Sign to use for the mass difference ratio value, 1 or -1

        returns:
            the reaction rate of n -> p
        """

        if q_sign not in [1, -1]:
            raise ValueError("q_sign must be 1 or -1")

        b = 5.93
        Z = b / T_9
        Z_nu = b / (self._T_nu(T_9))

        def I1(x):
            """Integrand of the first integral in equation (12) of the project."""
            return (
                (x + q_sign * self.q)
                * (x + q_sign * self.q)
                * np.sqrt(x * x - 1)
                * x
                / ((1 + np.exp(x * Z)) * (1 + np.exp(Z_nu * -(x + q_sign * self.q))))
            )

        def I2(x):
            """Integrand of the second integral in equation (12) of the project."""
            return (
                (x - q_sign * self.q)
                * (x - q_sign * self.q)
                * np.sqrt(x * x - 1)
                * x
                / ((1 + np.exp(-x * Z)) * (1 + np.exp(Z_nu * (x - q_sign * self.q))))
            )

        return 1 / self.tau * (quad(I1, 1, np.inf)[0] + quad(I2, 1, np.inf)[0])

    def _lambda_pn(self, T_9: float) -> float:
        """Describes the reaction rate of p -> n, equation (13) of the project,
        described in a.3 table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]

        returns:
            the reaction rate of p -> n
        """
        return self._lambda_np(T_9, -1)

    @lru_cache
    def _T_nu(self, T: float) -> float:
        """Neutrino temperature conversion

        arguments:
            T: temperature [K]

        returns:
            the neutrino temperature
        """
        return (4 / 11) ** (1 / 3) * T

    def get_weak_rates(self, T_9: float) -> tuple[float, float]:
        """Get the weak interaction rates for n -> p and p -> n.

        arguments:
            T_9: temperature [10^9 K]

        returns:
            the reaction rates of n -> p and p -> n
        """
        return self._lambda_np(T_9), self._lambda_pn(T_9)

    def get_np_to_D(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of n + p -> D + gamma and the reverse,
        described in b.1 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density [g/cm^3]

        returns:
            the reaction rate of n + p -> D + gamma, and the reverse
        """

        rate_np_to_Dgamma = 2.5e4 * rho_b  # b.1
        rate_D_to_np = (
            4.68e9 * rate_np_to_Dgamma / rho_b * T_9 ** (3 / 2) * np.exp(-25.82 / T_9)
        )

        return rate_np_to_Dgamma, rate_D_to_np

    def get_nD_to_T(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of n + D -> T + gamma, and the reverse,
        described in b.3 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of n + p -> D + gamma, and the reverse
        """

        rate_nD_to_Tgamma = rho_b * (75.5 + 1250 * T_9)
        rate_T_to_nD = (
            1.63e10 * rate_nD_to_Tgamma / rho_b * T_9 ** (3 / 2) * np.exp(-72.62 / T_9)
        )

        return rate_nD_to_Tgamma, rate_T_to_nD

    def get_DD_to_pT(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of D + D -> p + T, and the reverse,
        described in b.8 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of D + D -> p + T, and the reverse
        """

        rate_DD_to_pT = (
            3.9e8
            * rho_b
            * T_9 ** (-2 / 3)
            * np.exp(-4.26 * T_9 ** (-1 / 3))
            * (1 + 0.0979 * T_9 ** (1 / 3) + 0.642 * T_9 ** (2 / 3) + 0.440 * T_9)
        )

        rate_pT_to_DD = 1.73 * rate_DD_to_pT * np.exp(-47.80 / T_9)

        return rate_DD_to_pT, rate_pT_to_DD

    def get_pD_to_He3(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (p + D <-> He3 + gamma)
        described in b.2 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (p + D -> He3 + gamma), and the reverse
        """

        rate_pD_to_He3 = (
            2.23e3
            * rho_b
            * T_9 ** (-2 / 3)
            * np.exp(-3.72 * T_9 ** (-1 / 3))
            * (1 + 0.112 * T_9 ** (1 / 3) + 33.8 * T_9 ** (2 / 3) + 2.65 * T_9)
        )

        rate_He3_to_pD = (
            1.63e10 * rate_pD_to_He3 / rho_b * T_9 ** (3 / 2) * np.exp(-63.75 / T_9)
        )

        return rate_pD_to_He3, rate_He3_to_pD

    def get_nHe3_to_pT(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (n + He3 <-> p + T)
        described in b.4 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (n + He3 -> p + T), and the reverse
        """

        rate_nHe3_to_pT = 7.06e8 * rho_b

        rate_pT_to_nHe3 = rate_nHe3_to_pT * np.exp(-8.864 / T_9)

        return rate_nHe3_to_pT, rate_pT_to_nHe3

    def get_DD_to_nHe3(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (D + D <-> n + He3),
        described in b.7 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (D + D -> n + He3), and the reverse
        """

        rate_DD_to_nHe3 = (
            3.9e8
            * rho_b
            * T_9 ** (-2 / 3)
            * np.exp(-4.26 * T_9 ** (-1 / 3))
            * (1 + 0.0979 * T_9 ** (1 / 3) + 0.642 * T_9 ** (2 / 3) + 0.440 * T_9)
        )

        rate_nHe3_to_DD = 1.73 * rate_DD_to_nHe3 * np.exp(-37.94 / T_9)

        return rate_DD_to_nHe3, rate_nHe3_to_DD
