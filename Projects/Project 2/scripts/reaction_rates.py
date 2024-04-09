from functools import lru_cache

from scipy.integrate import quad
import numpy as np


class ReactionRates:
    """Reaction rate class"""

    def init(self, tau: float = 1700) -> None:
        """Initializes the class with a given neutron lifetime.

        arguments:
            tau: free neutron decay time [s]
        """
        self.tau = tau
        self.q = 2.53  # mass difference ratio [(m_n - m_p) / m_e]

    @lru_cache
    def _gamma_np(self, T_9: float, q_sign: int = 1) -> float:
        """Describes the reaction rate of n -> p, equation (12) of the project.

        arguments:
            T_9: temperature [10^9 K]
            q: Sign to use for the mass difference ratio value, 1 or -1

        returns:
            the reaction rate of n -> p
        """

        if q_sign not in [1, -1]:
            raise ValueError("q_sign must be 1 or -1")

        a = 5.93
        Z = a / T_9
        Z_nu = a / (self._T_nu(T_9))

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

    def _gamma_pn(self, T: float) -> float:
        """Describes the reaction rate of p -> n, equation (13) of the project.

        arguments:
            T: temperature [K]

        returns:
            the reaction rate of p -> n
        """
        return self._gamma_np(T, -q)

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
        return self._gamma_np(T_9), self._gamma_pn(T_9)
