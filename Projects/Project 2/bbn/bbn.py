"""Main Big Bang Nucleosynthesis module"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from reaction_rates import ReactionRates
from background import Background

# Ignore overflow runtimewarning
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Directory to save figures
FIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "figures"))
# Create this directory if it doesn't already exist
if not os.path.exists(FIG_DIR):
    os.makedirs(FIG_DIR)


class BBN:
    """
    General Big Bang nucleosynthesis class solving the Boltzmann equations for a collection
    of particle species in the early universe .
    """

    def __init__(self, N_species: int = 2, **background_kwargs) -> None:
        """
        Initializes the class with the given parameters.

        arguments:
            N_species: the number of particle species in the interval [2, 8]
            background_kwargs: keyword arguments to pass to the background class
        """
        if not (2 <= N_species <= 8):
            raise ValueError(
                "The number of interacting particle species must be between 2 and 8."
            )

        # Constants
        self.N_species = N_species
        self.species_labels = [  # species names
            "n",
            "p",
            "D",
            "T",
            "He3",
            "He4",
            "Li7",
            "Be7",
        ]

        self.mass_numbers = [1, 1, 2, 3, 3, 4, 7, 7]  # The particle atomic numbers

        # Initialize the reaction rates and background
        self.RR = ReactionRates()
        self.background = Background(**background_kwargs)

        # Initialize empty variables
        self.Y = None
        self.T = None

    def _get_ode_system(self, lnT: float, Y: np.ndarray) -> list[float]:
        """The right hand side of the coupled ODE system, for Y_i ODE's.

        arguments:
            lnT: logarithmic temperature
            Y: array with initial values of Y_i like so [Y_n, Y_p, ...]

        returns:
            The right hand sides of the ode's
        """

        # Initialize differential for each species i
        dY = np.zeros_like(Y)
        # dY[0] = dY_n, dY[1] = dY_p, ...

        T = np.exp(lnT)
        T_9 = T / 1e9

        # Neutrons and protons are always included
        Y_n = Y[0]
        Y_p = Y[1]

        # (n <-> p) [a.1-3]
        rate_n_to_p, rate_p_to_n = self.RR.get_weak_rates(T_9)

        # Change for left hand side reaction
        change_lhs = Y_p * rate_p_to_n - Y_n * rate_n_to_p
        # right hand side reaction change is equal, but opposite sign
        change_rhs = -change_lhs

        # Update neutron and proton ODE's
        dY[0] += change_lhs
        dY[1] += change_rhs

        if self.N_species > 2:  # Include deuterium
            Y_D = Y[2]
            rho_b = self.background.rho_b(T)  # calculate baryon density

            # (n+p <-> D + gamma) (b.1)
            rate_np_to_D, rate_D_to_np = self.RR.get_np_to_D(T_9, rho_b)

            # new changes
            change_lhs = Y_D * rate_D_to_np - Y_n * Y_p * rate_np_to_D
            change_rhs = -change_lhs

            # Update ODE's
            dY[0] += change_lhs
            dY[1] += change_lhs
            dY[2] += change_rhs

        if self.N_species > 3:  # include trituim
            Y_T = Y[3]

            # (n+D <-> T+gamma) (b.3)
            rate_nD_to_T, rate_T_to_nD = self.RR.get_nD_to_T(T_9, rho_b)

            # new changes
            change_lhs = Y_T * rate_T_to_nD - Y_n * Y_D * rate_nD_to_T
            change_rhs = -change_lhs

            # Update ODE's
            dY[0] += change_lhs
            dY[2] += change_lhs
            dY[3] += change_rhs

            # (D+D <-> p+T) (b.8)
            rate_DD_to_pT, rate_pT_to_DD = self.RR.get_DD_to_pT(T_9, rho_b)

            # new changes
            a = Y_p * Y_T * rate_pT_to_DD
            b = Y_D * Y_D * rate_DD_to_pT
            change_lhs = 2 * a - b
            change_rhs = 0.5 * b - a

            # Update ODE's
            dY[2] += change_lhs
            dY[1] += change_rhs
            dY[3] += change_rhs

        if self.N_species > 4:  # include helium-3
            Y_He3 = Y[4]

            # (p + D <-> He3 + gamma) (b.2)
            rate_pD_to_He3, rate_He3_to_pD = self.RR.get_pD_to_He3(T_9, rho_b)

            # new changes
            change_lhs = Y_He3 * rate_He3_to_pD - Y_p * Y_D * rate_pD_to_He3
            change_rhs = -change_lhs

            # Update ODE's
            dY[0] += change_lhs
            dY[2] += change_lhs

            dY[4] += change_rhs

            # (n + He3 <-> p + T) (b.4)
            rate_nHe3_to_pT, rate_pT_to_nHe3 = self.RR.get_nHe3_to_pT(T_9, rho_b)

            # new changes
            change_lhs = Y_p * Y_T * rate_pT_to_nHe3 - Y_n * Y_He3 * rate_nHe3_to_pT
            change_rhs = -change_lhs

            # Update ODE's
            dY[0] += change_lhs
            dY[4] += change_lhs
            dY[1] += change_rhs
            dY[3] += change_rhs

            # (D + D <-> n + He3) (b.7)
            rate_DD_to_nHe3, rate_nHe3_to_DD = self.RR.get_DD_to_nHe3(T_9, rho_b)

            # new changes
            a = Y_n * Y_He3 * rate_nHe3_to_DD
            b = Y_D * Y_D * rate_DD_to_nHe3
            change_lhs = 2 * a - b
            change_rhs = 0.5 * b - a

            # Update ODE's
            dY[2] += change_lhs
            dY[0] += change_rhs
            dY[4] += change_rhs

        if self.N_species > 5:  # include helium-4
            Y_He4 = Y[5]

            # (p + T <-> He4 + gamma) (b.5)
            rate_pT_to_He4, rate_He4_to_pT = self.RR.get_pT_to_He4(T_9, rho_b)

            # new changes
            change_lhs = Y_He4 * rate_He4_to_pT - Y_p * Y_T * rate_pT_to_He4
            change_rhs = -change_lhs

            # Update ODE's
            dY[1] += change_lhs
            dY[3] += change_lhs
            dY[5] += change_rhs

            # (n + He3 <-> He4 + gamma) (b.6)
            rate_nHe3_to_He4, rate_He4_to_nHe3 = self.RR.get_nHe3_to_He4(T_9, rho_b)

            # new changes
            change_lhs = Y_He4 * rate_He4_to_nHe3 - Y_n * Y_He3 * rate_nHe3_to_He4
            change_rhs = -change_lhs

            # Update ODE's
            dY[0] += change_lhs
            dY[4] += change_lhs
            dY[5] += change_rhs

            # (D + D <-> He4 + gamma) (b.9)
            rate_DD_to_He4, rate_He4_to_DD = self.RR.get_DD_to_He4(T_9, rho_b)

            # new changes
            a = Y_He4 * rate_He4_to_DD
            b = Y_D * Y_D * rate_DD_to_He4
            change_lhs = 2 * a - b
            change_rhs = 0.5 * b - a

            # Update ODE's
            dY[2] += change_lhs
            dY[5] += change_lhs

            # (D + He3 <-> He4 + p) (b.10)
            rate_DHe3_to_He4p, rate_He4p_to_DHe3 = self.RR.get_DHe3_to_He4p(T_9, rho_b)

            # new changes
            change_lhs = (
                Y_He4 * Y_p * rate_He4p_to_DHe3 - Y_D * Y_He3 * rate_DHe3_to_He4p
            )
            change_rhs = -change_lhs

            # Update ODE's
            dY[2] += change_lhs
            dY[4] += change_lhs
            dY[1] += change_rhs
            dY[5] += change_rhs

            # (D + T <-> He4 + n) (b.11)
            rate_DT_to_He4n, rate_He4n_to_DT = self.RR.get_DT_to_He4n(T_9, rho_b)

            # new changes
            change_lhs = Y_He4 * Y_n * rate_He4n_to_DT - Y_D * Y_T * rate_DT_to_He4n
            change_rhs = -change_lhs

            # Update ODE's
            dY[2] += change_lhs
            dY[3] += change_lhs
            dY[5] += change_rhs
            dY[0] += change_rhs

            # (He3 + T <-> He4 + D) (b.15)
            rate_He3T_to_He4D, rate_He4D_to_He3T = self.RR.get_He3T_to_He4D(T_9, rho_b)

            # new changes
            change_lhs = (
                Y_He4 * Y_D * rate_He4D_to_He3T - Y_He3 * Y_T * rate_He3T_to_He4D
            )
            change_rhs = -change_lhs

            # Update ODE's
            dY[4] += change_lhs
            dY[3] += change_lhs
            dY[5] += change_rhs
            dY[2] += change_rhs

        if self.N_species > 6:  # include lithium-7
            Y_Li7 = Y[6]

            # (T + He4 <-> Li7 + gamma) (b.17)
            rate_THe4_to_Li7, rate_Li7_to_THe4 = self.RR.get_THe4_to_Li7(T_9, rho_b)

            # new changes
            change_lhs = Y_Li7 * rate_Li7_to_THe4 - Y_T * Y_He4 * rate_THe4_to_Li7
            change_rhs = -change_lhs

            # Update ODE's
            dY[3] += change_lhs
            dY[5] += change_lhs
            dY[6] += change_rhs

            # (p + Li7 <-> He4 + He4) (b.20)
            rate_pLi7_to_He4He4, rate_He4He4_to_pLi7 = self.RR.get_pLi7_to_He4He4(
                T_9, rho_b
            )

            # new changes
            a = Y_He4 * Y_He4 * rate_He4He4_to_pLi7
            b = Y_p * Y_Li7 * rate_pLi7_to_He4He4
            change_lhs = 0.5 * a - b
            change_rhs = 2 * b - a

            # Update ODE's
            dY[1] += change_lhs
            dY[6] += change_lhs
            dY[5] += change_rhs

        if self.N_species > 7:  # include beryllium-7
            Y_Be7 = Y[7]

            # (He3 + He4 <-> Be7 + gamma) (b.16)
            rate_He3He4_to_Be7, rate_Be7_to_He3He4 = self.RR.get_He3He4_to_Be7(
                T_9, rho_b
            )

            # new changes
            change_lhs = Y_Be7 * rate_Be7_to_He3He4 - Y_He3 * Y_He4 * rate_He3He4_to_Be7
            change_rhs = -change_lhs

            # Update ODE's
            dY[4] += change_lhs
            dY[5] += change_lhs
            dY[7] += change_rhs

            # (n + Be7 <-> p + Li7) (b.18)
            rate_nBe7_to_pLi7, rate_pLi7_to_nBe7 = self.RR.get_nBe7_to_pLi7(T_9, rho_b)

            # new changes
            change_lhs = (
                Y_p * Y_Li7 * rate_pLi7_to_nBe7 - Y_n * Y_Be7 * rate_nBe7_to_pLi7
            )
            change_rhs = -change_lhs

            # Update ODE's
            dY[0] += change_lhs
            dY[7] += change_lhs
            dY[1] += change_rhs
            dY[6] += change_rhs

            # (n + Be7 <-> He4 + He4) (b.21)
            rate_nBe7_to_He4He4, rate_He4He4_to_nBe7 = self.RR.get_nBe7_to_He4He4(
                T_9, rho_b
            )

            # new changes
            a = Y_He4 * Y_He4 * rate_He4He4_to_nBe7
            b = Y_n * Y_Be7 * rate_nBe7_to_He4He4
            change_lhs = 0.5 * a - b
            change_rhs = 2 * b - a

            # Update ODE's
            dY[0] += change_lhs
            dY[7] += change_lhs
            dY[5] += change_rhs

        return -dY / self.background.H(T)  # Multiply every term by -1/H

    def _Y_n_equil(self, T: float) -> float:
        """Thermal equilibrium value of relative number density of neutrons Y_n, equation (16) of the project.

        arguments:
            T: temperature [K]

        returns:
            the thermal equilibrium value of Y_n
        """
        return 1 / (
            1
            + np.exp(
                (self.background.const.m_n - self.background.const.m_p)
                * self.background.const.c
                * self.background.const.c
                / (self.background.const.k * T)
            )
        )

    def _Y_p_equil(self, T: float) -> float:
        """Thermal equilibrium value of relative number density of protons Y_p, equation (17) of the project.

        arguments:
            T: temperature [K]

        returns:
            the thermal equilibrium value of Y_n
        """
        return 1 - self._Y_n_equil(T)

    def get_initial_conditions(self, T_i: float) -> np.ndarray:
        """Returns the initial conditions for the ODE system.

        arguments:
            T_i: initial temperature [K]

        returns:
            the initial conditions array
        """

        Y_i = np.zeros(self.N_species)  # init all species to zero

        # Initialize weak equilibrium values
        Y_i[0] = self._Y_n_equil(T_i)
        Y_i[1] = self._Y_p_equil(T_i)

        # The rest of the species are set to zero initially
        return Y_i

    def solve_ode_system(
        self, T_i: float, T_f: float, n_points: int = 1000, tol: float = 1e-12
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solves the ode system of the equations of motion for Y_n and Y_p

        arguments:
            T_i: initial temperature [K]
            T_f: final temperature [K]
            n_points: number of points in the solution array

        returns:
            the logarithmic temperature array lnT, and the solution array Y
        """

        # Initial conditions
        Y = self.get_initial_conditions(T_i)

        # Solve the ODE system
        sol = solve_ivp(
            self._get_ode_system,
            [np.log(T_i), np.log(T_f)],
            Y,
            method="Radau",
            rtol=tol,
            atol=tol,
            dense_output=True,
        )

        # Store values and also return them
        lnT = np.linspace(sol.t[0], sol.t[-1], n_points)
        self.Y = sol.sol(lnT)
        self.T = np.exp(lnT)

        return self.T, self.Y

    def plot_mass_fractions(
        self,
        filename: str = None,
        figsize: tuple[int, int] = (7, 5),
        ymin: float = 1e-3,
        ymax: float = 2.0,
    ) -> None:
        """Plots the mass fractions A_i*Y_i for each species, as a function of logarithmic temperature ln(T)

        arguments:
            filename: the filename to save the plot figure
            figsize: the plot figure size
            ymin: the minimum y-axis value
            ymax: the maximum y-axis value

        returns:
            None
        """

        # If ode system has not been solved already, return early
        if self.T is None:
            print("Warning: cannot plot before solving the ODE system.")
            return

        fig, ax = plt.subplots(figsize=figsize)

        total = 0  # total sum of mass fraction
        for i, y in enumerate(self.Y):
            # Mass fraction
            A = y * self.mass_numbers[i]

            # Plot mass fraction of species i
            ax.loglog(self.T, A, label=self.species_labels[i])

            # Add to total sum
            total += A

        # Plot thermal equilibrium value of neutron and proton
        ax.loglog(
            self.T,
            self._Y_n_equil(self.T) * self.mass_numbers[0],
            color="C0",
            linestyle=":",
        )
        ax.loglog(
            self.T,
            self._Y_p_equil(self.T) * self.mass_numbers[1],
            color="C1",
            linestyle=":",
        )

        # Finally plot the sum of the relative densities, which should always be equal to one
        ax.loglog(self.T, total, "k:", label=r"$\sum_i A_iY_i$")

        # Plot settings
        plt.gca().invert_xaxis()  # invert x-axis
        plt.xlabel("T [K]")
        plt.ylabel(r"Mass fraction $A_iY_i$")

        # Set y-axis limits
        plt.ylim(bottom=ymin, top=ymax)

        # Add 10^0=1 to y-ticks if it is not already there (to better show the mass fraction sum)
        ticks = list(plt.yticks()[0])
        if 1 not in ticks:
            i = 0
            while ticks[i] < 1:
                i += 1
            ticks.insert(i, 1)
        plt.yticks(ticks)

        # Set y-axis limits
        plt.ylim(bottom=ymin, top=ymax)

        plt.legend()
        plt.grid()
        plt.title("Mass fractions of particles species")

        # Save figure if filename is given
        if filename:
            plt.savefig(filename)
        else:
            plt.show()


if __name__ == "__main__":
    ### DIRECT USAGE EXAMPLE ###

    # Variables
    N_species = 2  # number of interacting atom species
    N_eff = 3  # effective number of neutrino species
    T_i = 1e11  # initial temperature [K]
    T_f = 1e8  # final temperature [K]
    n_points = 1000  # number of points for plotting
    unit = "cgs"  # unit system to use
    filename = os.path.join(FIG_DIR, "example_problem_f.png")

    # Initialize
    bbn = BBN(N_species, N_eff=N_eff, unit=unit)

    # Solve ode
    bbn.solve_ode_system(T_i, T_f, n_points)

    # Plot
    bbn.plot_mass_fractions(filename)
