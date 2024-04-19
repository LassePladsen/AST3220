"""Reaction rates for the early universe."""

from functools import lru_cache

from scipy.integrate import quad
import numpy as np


# These three functions are extracted from the class to avoid calculating the integral
# every time a new class instance is created, like in the later problems


@lru_cache
def T_nu(T: float) -> float:
    """Neutrino temperature conversion

    arguments:
        T: temperature [K]

    returns:
        the neutrino temperature
    """
    return (4 / 11) ** (1 / 3) * T


@lru_cache
def lambda_n_top(T_9: float, q_sign: int = 1) -> float:
    """Describes the reaction rate of n -> p, equation (12) of the project,
    described in a.3 table 2 of the project reference material:
    https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

    arguments:
        T_9: temperature [10^9 K]
        q: Sign to use for the mass difference ratio value, 1 or -1

    returns:
        the reaction rate of n -> p [1/s]
    """

    if q_sign not in [1, -1]:
        raise ValueError("q_sign must be 1 or -1")

    b = 5.93
    Z = b / T_9
    Z_nu = b / (T_nu(T_9))

    q = 2.53  # mass difference ratio [(m_n - m_p) / m_e]

    tau = 1700  # free neutron decay time [s]

    def I1(x):
        """Integrand of the first integral in equation (12) of the project."""
        a = x + q_sign * q
        return (
            a
            * a
            * np.sqrt(x * x - 1)
            * x
            / ((1 + np.exp(x * Z)) * (1 + np.exp(-Z_nu * a)))
        )

    def I2(x):
        """Integrand of the second integral in equation (12) of the project."""
        a = x - q_sign * q
        return (
            a
            * a
            * np.sqrt(x * x - 1)
            * x
            / ((1 + np.exp(-x * Z)) * (1 + np.exp(Z_nu * a)))
        )

    return 1 / tau * (quad(I1, 1, np.inf)[0] + quad(I2, 1, np.inf)[0])


def lambda_p_to_n(T_9: float) -> float:
    """Describes the reaction rate of p -> n, equation (13) of the project,
    described in a.3 table 2 of the project reference material:
    https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

    arguments:
        T_9: temperature [10^9 K]

    returns:
        the reaction rate of p -> n [1/s]
    """
    return lambda_n_top(T_9, -1)


class ReactionRates:
    """Class describing reaction rates between the particle species in the early universe."""

    def get_weak_rates(self, T_9: float) -> tuple[float, float]:
        """Get the weak interaction rates for n -> p and p -> n.

        arguments:
            T_9: temperature [10^9 K]

        returns:
            the reaction rates of n -> p and p -> n  [1/s]
        """
        return lambda_n_top(T_9), lambda_p_to_n(T_9)

    def get_np_to_D(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of n + p -> D + gamma and the reverse,
        described in b.1 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density [g/cm^3]

        returns:
            the reaction rate of n + p -> D + gamma, and the reverse  [1/s]
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
            the reaction rate of n + p -> D + gamma, and the reverse  [1/s]
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
            the reaction rate of D + D -> p + T, and the reverse [1/s]
        """

        rate_DD_to_pT = (
            3.9e8
            * rho_b
            * T_9 ** (-2 / 3)
            * np.exp(-4.26 * T_9 ** (-1 / 3))
            * (1 + 0.0979 * T_9 ** (1 / 3) + 0.642 * T_9 ** (2 / 3) + 0.440 * T_9)
        )

        rate_pT_to_DD = 1.73 * rate_DD_to_pT * np.exp(-46.80 / T_9)

        return rate_DD_to_pT, rate_pT_to_DD

    def get_pD_to_He3(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (p + D <-> He3 + gamma)
        described in b.2 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (p + D -> He3 + gamma), and the reverse [1/s]
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
            the reaction rate of (n + He3 -> p + T), and the reverse [1/s]
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
            the reaction rate of (D + D -> n + He3), and the reverse [1/s]
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

    def get_pT_to_He4(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (p + T <-> He4 + gamma),
        described in b.5 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (p + T -> He4 + gamma), and the reverse [1/s]
        """

        rate_pT_to_He4 = (
            2.87e4
            * rho_b
            * T_9 ** (-2 / 3)
            * np.exp(-3.87 / T_9)
            * (
                1
                + 0.108 * T_9 ** (1 / 3)
                + 0.466 * T_9 ** (2 / 3)
                + 0.352 * T_9
                + 0.300 * T_9 ** (4 / 3)
                + 0.576 * T_9 ** (5 / 3)
            )
        )

        rate_He4_to_pT = (
            2.59e10 * rate_pT_to_He4 / rho_b * T_9 ** (3 / 2) * np.exp(-229.9 / T_9)
        )

        return rate_pT_to_He4, rate_He4_to_pT

    def get_nHe3_to_He4(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (n + He3 <-> He4 + gamma),
        described in b.6 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (n + He3 -> He4 + gamma), and the reverse [1/s]
        """

        rate_nHe3_to_He4 = 6e3 * rho_b * T_9

        rate_He4_to_nHe3 = (
            2.6e10 * rate_nHe3_to_He4 / rho_b * T_9 ** (3 / 2) * np.exp(-238.8 / T_9)
        )

        return rate_nHe3_to_He4, rate_He4_to_nHe3

    def get_DD_to_He4(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (D + D <-> He4 + gamma),
        described in b.9 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (D + D -> He4 + gamma), and the reverse [1/s]
        """

        rate_DD_to_He4 = (
            24.1
            * rho_b
            * T_9 ** (-2 / 3)
            * np.exp(-4.26 / T_9 ** (-1 / 3))
            * (
                T_9 ** (2 / 3)
                + 0.685 * T_9
                + 0.152 * T_9 ** (4 / 3)
                + 0.265 * T_9 ** (5 / 3)
            )
        )

        rate_He4_to_DD = (
            4.5e10 * rate_DD_to_He4 / rho_b * T_9 ** (3 / 2) * np.exp(-276.7 / T_9)
        )

        return rate_DD_to_He4, rate_He4_to_DD

    def get_DHe3_to_He4p(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (D + He3 <-> He4 + p),
        described in b.10 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (D + He3 -> He4 + p), and the reverse [1/s]
        """

        rate_DHe3_to_He4p = 2.6e9 * rho_b * T_9 ** (-3 / 2) * np.exp(-2.99 / T_9)

        rate_He4p_to_DHe3 = 5.5 * rate_DHe3_to_He4p * np.exp(-213 / T_9)

        return rate_DHe3_to_He4p, rate_He4p_to_DHe3

    def get_DT_to_He4n(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (D + T <-> He4 + n),
        described in b.11 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (D + T -> He4 + n), and the reverse [1/s]
        """

        rate_DT_to_He4n = 1.38e9 * rho_b * T_9 ** (-3 / 2) * np.exp(-0.745 / T_9)

        rate_He4n_to_DT = 5.5 * rate_DT_to_He4n * np.exp(-204.1 / T_9)

        return rate_DT_to_He4n, rate_He4n_to_DT

    def get_He3T_to_He4D(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (He3 + T <-> He4 + D),
        described in b.15 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (He3 + T -> He4 + D), and the reverse [1/s]
        """

        rate_He3T_to_He4D = (
            3.88e9
            * rho_b
            * T_9 ** (-2 / 3)
            * np.exp(-7.72 * T_9 ** (-1 / 3))
            * (1 + 0.0540 * T_9 ** (1 / 3))
        )

        rate_He4D_to_He3T = 1.59 * rate_He3T_to_He4D * np.exp(-166.2 / T_9)

        return rate_He3T_to_He4D, rate_He4D_to_He3T

    def get_THe4_to_Li7(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (T + He4 <-> Li7 + gamma),
        described in b.17 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (T + He4 -> Li7 + gamma), and the reverse [1/s]
        """

        rate_THe4_to_Li7 = (
            5.28e5
            * rho_b
            * T_9 ** (-2 / 3)
            * np.exp(-8.08 * T_9 ** (-1 / 3))
            * (1 + 0.0516 * T_9 ** (1 / 3))
        )

        rate_Li7_to_THe4 = (
            1.12e10 * rate_THe4_to_Li7 / rho_b * T_9 ** (3 / 2) * np.exp(-28.63 / T_9)
        )

        return rate_THe4_to_Li7, rate_Li7_to_THe4

    def get_pLi7_to_He4He4(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (p + Li7 <-> He4 + He4),
        described in b.20 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (p + Li7 -> He4 + He4), and the reverse [1/s]
        """

        rate_pLi7_to_He4He4 = (
            1.42e9
            * rho_b
            * T_9 ** (-2 / 3)
            * np.exp(-8.47 * T_9 ** (-1 / 3))
            * (1 + 0.0493 * T_9 ** (1 / 3))
        )

        rate_He4He4_to_pLi7 = 4.64 * rate_pLi7_to_He4He4 * np.exp(-201.3 / T_9)

        return rate_pLi7_to_He4He4, rate_He4He4_to_pLi7

    def get_nBe7_to_pLi7(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (n + Be7 <-> p + Li7),
        described in b.18 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (n + Be7 -> p + Li7), and the reverse [1/s]
        """

        rate_nBe7_to_pLi7 = 6.74e9 * rho_b

        rate_pLi7_to_nBe7 = rate_nBe7_to_pLi7 * np.exp(-19.07 / T_9)

        return rate_nBe7_to_pLi7, rate_pLi7_to_nBe7

    def get_nBe7_to_He4He4(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (n + Be7 <-> He4 + He4),
        described in b.21 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (n + Be7 -> He4 + He4), and the reverse [1/s]
        """

        rate_nBe7_to_He4He4 = 1.2e7 * rho_b * T_9

        rate_He4He4_to_nBe7 = 4.64 * rate_nBe7_to_He4He4 * np.exp(-220.4 / T_9)

        return rate_nBe7_to_He4He4, rate_He4He4_to_nBe7

    def get_He3He4_to_Be7(self, T_9: float, rho_b: float) -> tuple[float, float]:
        """Describes the reaction rate of (He3 + He4 <-> Be7 + gamma),
        described in b.16 of table 2 of the project reference material:
        https://www.uio.no/studier/emner/matnat/astro/AST3220/v24/undervisningsmateriale/wagoner-fowler-hoyle.pdf

        arguments:
            T_9: temperature [10^9 K]
            rho_b: baryon density

        returns:
            the reaction rate of (He3 + He4 -> Be7 + gamma), and the reverse [1/s]
        """

        rate_He3He4_to_Be7 = (
            4.8e6
            * rho_b
            * T_9 ** (-2 / 3)
            * np.exp(-12.8 * T_9 ** (-1 / 3))
            * (
                1
                + 0.0326 * T_9 ** (1 / 3)
                - 0.219 * T_9 ** (2 / 3)
                + 0.0499 * T_9
                + 0.0258 * T_9 ** (4 / 3)
                + 0.0150 * T_9 ** (5 / 3)
            )
        )

        rate_Be7_to_He3He4 = (
            1.12e10 * rate_He3He4_to_Be7 / rho_b * T_9 ** (3 / 2) * np.exp(-18.42 / T_9)
        )

        return rate_He3He4_to_Be7, rate_Be7_to_He3He4


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, sharex=True)
    T9 = np.logspace(11, 7, 200) / 1e9
    n = np.zeros_like(T9)
    p = np.zeros_like(T9)
    for i, Ti in enumerate(T9):
        n[i] = lambda_n_top(Ti)
        p[i] = lambda_p_to_n(Ti)
    T = T9 * 1e9
    axs[0].loglog(T, n)
    axs[0].set_title("n -> p")
    axs[0].grid(True)
    axs[1].loglog(T, p)
    axs[1].set_title("p -> n")
    axs[1].grid(True)
    fig.supxlabel("T [K]")
    fig.supylabel("Reaction rate")
    plt.gca().invert_xaxis()
    plt.show()
