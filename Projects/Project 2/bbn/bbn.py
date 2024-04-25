"""Main Big Bang Nucleosynthesis module"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .reaction_rates import ReactionRates
from .background import Background
from .stats import bayesian_probability

# Ignore overflow runtimewarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Directory to save figures
FIG_DIR = Path(__file__).parents[1] / "figures"


class BBN:
    """
    General Big Bang nucleosynthesis class solving the Boltzmann equations for a collection
    of particle species in the early universe .
    """

    # Class constants
    SPECIES_LABELS = (  # species names
        "n",
        "p",
        "D",
        "T",
        "He3",
        "He4",
        "Li7",
        "Be7",
    )

    # Color name array for consistent colors when plotting
    COLORS = (
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
    )

    MASS_NUMBERS = (1, 1, 2, 3, 3, 4, 7, 7)  # The particle atomic numbers

    # Observed values for relic abundances Y_i/Y_p
    D_ABUNDANCE = 2.57e-5  # Y_D/Y_p
    D_ABUNDANCE_ERR = 0.03e-5  # error in Y_D/Y_p
    LI7_ABUNDANCE = 1.6e-10
    LI7_ABUNDANCE_ERR = 0.3e-10

    # Observed value for mass fraction 4Y_He4
    HE4_MASS_FRAC = 0.254  # 4Y_He4
    HE4_MASS_FRAC_ERR = 0.003  # error in 4Y_He4

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

        # Initialize the number of species
        self.N_species = N_species

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

            Y_np = Y_n * Y_p
            # new changes
            change_lhs = Y_D * rate_D_to_np - Y_np * rate_np_to_D
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
            Y_nD = Y_n * Y_D
            change_lhs = Y_T * rate_T_to_nD - Y_nD * rate_nD_to_T
            change_rhs = -change_lhs

            # Update ODE's
            dY[0] += change_lhs
            dY[2] += change_lhs
            dY[3] += change_rhs

            # (D+D <-> p+T) (b.8)
            rate_DD_to_pT, rate_pT_to_DD = self.RR.get_DD_to_pT(T_9, rho_b)

            # new changes
            Y_pT = Y_p * Y_T
            Y_DD = Y_D * Y_D
            a = Y_pT * rate_pT_to_DD
            b = Y_DD * rate_DD_to_pT
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
            Y_pD = Y_p * Y_D
            change_lhs = Y_He3 * rate_He3_to_pD - Y_pD * rate_pD_to_He3
            change_rhs = -change_lhs

            # Update ODE's
            dY[1] += change_lhs
            dY[2] += change_lhs
            dY[4] += change_rhs

            # (n + He3 <-> p + T) (b.4)
            rate_nHe3_to_pT, rate_pT_to_nHe3 = self.RR.get_nHe3_to_pT(T_9, rho_b)

            # new changes
            Y_nHe3 = Y_n * Y_He3
            change_lhs = Y_pT * rate_pT_to_nHe3 - Y_nHe3 * rate_nHe3_to_pT
            change_rhs = -change_lhs

            # Update ODE's
            dY[0] += change_lhs
            dY[4] += change_lhs
            dY[1] += change_rhs
            dY[3] += change_rhs

            # (D + D <-> n + He3) (b.7)
            rate_DD_to_nHe3, rate_nHe3_to_DD = self.RR.get_DD_to_nHe3(T_9, rho_b)

            # new changes
            a = Y_nHe3 * rate_nHe3_to_DD
            b = Y_DD * rate_DD_to_nHe3
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
            change_lhs = Y_He4 * rate_He4_to_pT - Y_pT * rate_pT_to_He4
            change_rhs = -change_lhs

            # Update ODE's
            dY[1] += change_lhs
            dY[3] += change_lhs
            dY[5] += change_rhs

            # (n + He3 <-> He4 + gamma) (b.6)
            rate_nHe3_to_He4, rate_He4_to_nHe3 = self.RR.get_nHe3_to_He4(T_9, rho_b)

            # new changes
            change_lhs = Y_He4 * rate_He4_to_nHe3 - Y_nHe3 * rate_nHe3_to_He4
            change_rhs = -change_lhs

            # Update ODE's
            dY[0] += change_lhs
            dY[4] += change_lhs
            dY[5] += change_rhs

            # (D + D <-> He4 + gamma) (b.9)
            rate_DD_to_He4, rate_He4_to_DD = self.RR.get_DD_to_He4(T_9, rho_b)

            # new changes
            a = Y_He4 * rate_He4_to_DD
            b = Y_DD * rate_DD_to_He4
            change_lhs = 2 * a - b
            change_rhs = 0.5 * b - a

            # Update ODE's
            dY[2] += change_lhs
            dY[5] += change_rhs

            # (D + He3 <-> He4 + p) (b.10)
            rate_DHe3_to_He4p, rate_He4p_to_DHe3 = self.RR.get_DHe3_to_He4p(T_9, rho_b)

            # new changes
            Y_He4p = Y_He4 * Y_p
            Y_DHe3 = Y_D * Y_He3
            change_lhs = Y_He4p * rate_He4p_to_DHe3 - Y_DHe3 * rate_DHe3_to_He4p
            change_rhs = -change_lhs

            # Update ODE's
            dY[2] += change_lhs
            dY[4] += change_lhs
            dY[5] += change_rhs
            dY[1] += change_rhs

            # (D + T <-> He4 + n) (b.11)
            rate_DT_to_He4n, rate_He4n_to_DT = self.RR.get_DT_to_He4n(T_9, rho_b)

            # new changes
            Y_He4n = Y_He4 * Y_n
            Y_DT = Y_D * Y_T
            change_lhs = Y_He4n * rate_He4n_to_DT - Y_DT * rate_DT_to_He4n
            change_rhs = -change_lhs

            # Update ODE's
            dY[2] += change_lhs
            dY[3] += change_lhs
            dY[5] += change_rhs
            dY[0] += change_rhs

            # (He3 + T <-> He4 + D) (b.15)
            rate_He3T_to_He4D, rate_He4D_to_He3T = self.RR.get_He3T_to_He4D(T_9, rho_b)

            # new changes
            Y_He4D = Y_He4 * Y_D
            Y_He3T = Y_He3 * Y_T
            change_lhs = Y_He4D * rate_He4D_to_He3T - Y_He3T * rate_He3T_to_He4D
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
            Y_THe4 = Y_T * Y_He4
            change_lhs = Y_Li7 * rate_Li7_to_THe4 - Y_THe4 * rate_THe4_to_Li7
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
            Y_He4He4 = Y_He4 * Y_He4
            Y_pLi7 = Y_p * Y_Li7
            a = Y_He4He4 * rate_He4He4_to_pLi7
            b = Y_pLi7 * rate_pLi7_to_He4He4
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
            Y_He3He4 = Y_He3 * Y_He4
            change_lhs = Y_Be7 * rate_Be7_to_He3He4 - Y_He3He4 * rate_He3He4_to_Be7
            change_rhs = -change_lhs

            # Update ODE's
            dY[4] += change_lhs
            dY[5] += change_lhs
            dY[7] += change_rhs

            # (n + Be7 <-> p + Li7) (b.18)
            rate_nBe7_to_pLi7, rate_pLi7_to_nBe7 = self.RR.get_nBe7_to_pLi7(T_9, rho_b)

            # new changes
            Y_nBe7 = Y_n * Y_Be7
            change_lhs = Y_pLi7 * rate_pLi7_to_nBe7 - Y_nBe7 * rate_nBe7_to_pLi7
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
            a = Y_He4He4 * rate_He4He4_to_nBe7
            b = Y_nBe7 * rate_nBe7_to_He4He4
            change_lhs = 0.5 * a - b
            change_rhs = 2 * b - a

            # Update ODE's
            dY[0] += change_lhs
            dY[7] += change_lhs
            dY[5] += change_rhs

        return -dY / self.background.H(T)  # Multiply every term by -1/H

    def _get_n_equil(self, T: float) -> float:
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

    def _get_p_equil(self, T: float) -> float:
        """Thermal equilibrium value of relative number density of protons Y_p, equation (17) of the project.

        arguments:
            T: temperature [K]

        returns:
            the thermal equilibrium value of Y_n
        """
        return 1 - self._get_n_equil(T)

    def get_initial_conditions(self, T_i: float) -> np.ndarray:
        """Returns the initial conditions for the ODE system.

        arguments:
            T_i: initial temperature [K]

        returns:
            the initial conditions array
        """

        Y_i = np.zeros(self.N_species)  # init all species to zero

        # Initialize weak equilibrium values
        Y_i[0] = self._get_n_equil(T_i)
        Y_i[1] = self._get_p_equil(T_i)

        # The rest of the species are set to zero initially
        return Y_i

    def solve_BBN(
        self, T_i: float, T_f: float, n_points: int = 1000, tol: float = 1e-12
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solves the BBN ode system of the equations of motion for Y_i

        arguments:
            T_i: initial temperature [K]
            T_f: final temperature [K]
            n_points: number of points in the solution array

        returns:
            the temperature array T, and the solution array Y
        """

        # Solve the ODE system
        sol = solve_ivp(
            self._get_ode_system,
            [np.log(T_i), np.log(T_f)],  # the equations are defined over ln(T)
            y0=self.get_initial_conditions(T_i),
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
        plot_equilibrium: bool = True,
    ) -> None:
        """Plots the mass fractions A_i*Y_i for each species, as a function of logarithmic temperature ln(T)

        arguments:
            filename: the filename to save the plot figure, if none the figure is shown
            figsize: the plot figure size
            ymin: the minimum y-axis value
            ymax: the maximum y-axis value
            plot_equilibrium: whether to plot the thermal equilibrium values of neutron and proton

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
            mass_frac = y * self.MASS_NUMBERS[i]

            # Plot mass fraction of species i
            ax.loglog(self.T, mass_frac, label=self.SPECIES_LABELS[i])

            # Add to total sum
            total += mass_frac

        if plot_equilibrium:
            # Plot thermal equilibrium value of neutron and proton
            ax.loglog(
                self.T,
                self._get_n_equil(self.T) * self.MASS_NUMBERS[0],
                color=BBN.COLORS[0],
                linestyle=":",
            )
            ax.loglog(
                self.T,
                self._get_p_equil(self.T) * self.MASS_NUMBERS[1],
                color=BBN.COLORS[1],
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
        new_tick = 1
        if new_tick not in ticks:
            i = 0
            while ticks[i] < new_tick:
                i += 1
            ticks.insert(i, new_tick)
        plt.yticks(ticks)

        # Set the new configured y-axis limits
        plt.ylim(bottom=ymin, top=ymax)

        plt.legend()
        plt.grid(True)
        # Grid ticks inside, and on all sides
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        plt.title("Mass fractions of particles species")

        # Save figure if filename is given
        if filename:
            # Create fig dir directory if it doesn't already exist
            if not FIG_DIR.exists():
                FIG_DIR.mkdir()
            plt.savefig(filename)
        else:
            plt.show()

    @staticmethod
    def interpolate_Y_of_Omegab0(
        T_i: float,
        T_f: float,
        Omega_b0_vals: np.ndarray,
        Y_min: float = 1e-20,
    ) -> tuple[callable, callable, callable, callable]:
        """Static method. Interpolates the mass fractions Y_i as a function of Omega_b0 to give a smooth
        function graph. Interpolates in logspace (base e)

        arguments:
            T_i: initial temperature [K]
            T_f: final temperature [K]
            Omega_b0_vals: array of Omega_b0 values to interpolate
            Y_min: lower bound value for mass fractions, everything below this value is set to Y_min

        returns:
            Tuple containing the logspace interpolated mass fractions functions logY_i(logOmega_b0)
            for i = p, D, He3, He4, and Li7
        """
        Y_model = np.zeros((5, len(Omega_b0_vals)))
        for i, Omega_b0 in enumerate(Omega_b0_vals):
            # Initialize instance
            bbn = BBN(N_species=8, Omega_b0=Omega_b0)

            # Solve ode for this Omega_b0
            _, Y = bbn.solve_BBN(T_i, T_f)

            # Extract mass fraction values for final temperature
            Y = Y[:, -1]

            # T and Be7 decays to respectively He3 and Li7
            Y[4] += Y[3]
            Y[6] += Y[7]

            # Set lower bound for mass fractions
            Y[Y < Y_min] = Y_min

            # Extract mass fractions
            Y_p = Y[1]
            Y_D = Y[2]
            Y_He3 = Y[4]
            Y_He4 = Y[5]
            Y_Li7 = Y[6]
            Y = np.asarray([Y_p, Y_D, Y_He3, Y_He4, Y_Li7])

            # Append processed mass fractions
            Y_model[:, i] = Y

        # Interpolate each function in logspace
        kind = "cubic"
        Y_p, Y_D, Y_He3, Y_He4, Y_Li7 = Y_model

        Y_p_interp = interp1d(np.log(Omega_b0_vals), np.log(Y_p), kind=kind)
        Y_D_interp = interp1d(np.log(Omega_b0_vals), np.log(Y_D), kind=kind)
        Y_He3_interp = interp1d(np.log(Omega_b0_vals), np.log(Y_He3), kind=kind)
        Y_He4_interp = interp1d(np.log(Omega_b0_vals), np.log(Y_He4), kind=kind)
        Y_Li7_interp = interp1d(np.log(Omega_b0_vals), np.log(Y_Li7), kind=kind)

        return Y_p_interp, Y_D_interp, Y_He3_interp, Y_He4_interp, Y_Li7_interp

    @staticmethod
    def plot_relic_abundances_Omegab0(
        T_i: float,
        T_f: float,
        Omega_b0_vals: np.ndarray,
        Y_min: float = 1e-20,
        n_plot: int = 300,
        filename: str = None,
        figsize: tuple[int, int] = (7, 6),
    ) -> None:
        """Static method. Plots the relic abundances of elements Y_i/Y_p in the BBN process, as a function
        of the baryon density parameter Omega_b0, by interpolating. Also finds the most likely
        Omega_b0 value by using Bayesian probability.

        arguments:
            T_i: initial temperature [K]
            T_f: final temperature [K]
            Omega_b0_vals: array of Omega_b0 values used to calculate before interpolation
            Y_min: lower bound value for mass fractions, everything below is set to this
            n_plot: number of points to use for plotting, after the interpolation
            filename: the filename to save the plot figure, if none the figure is shown
            figsize: the plot figure size
        """

        # Solve ODE for the values and interpolate
        (
            Y_p_interp_func,
            Y_D_interp_func,
            Y_He3_interp_func,
            Y_He4_interp_func,
            Y_Li7_interp_func,
        ) = BBN.interpolate_Y_of_Omegab0(T_i, T_f, Omega_b0_vals, Y_min=Y_min)

        # Array to interpolate graph with
        Omega_b0_arr = np.logspace(
            np.log10(Omega_b0_vals[0]), np.log10(Omega_b0_vals[-1]), n_plot
        )
        log_Omega_b0_arr = np.log(Omega_b0_arr)

        # Interpolate with these values
        Y_p_interp = np.exp(Y_p_interp_func(log_Omega_b0_arr))
        Y_D_interp = np.exp(Y_D_interp_func(log_Omega_b0_arr))
        Y_He3_interp = np.exp(Y_He3_interp_func(log_Omega_b0_arr))
        Y_He4_interp = np.exp(Y_He4_interp_func(log_Omega_b0_arr))
        Y_Li7_interp = np.exp(Y_Li7_interp_func(log_Omega_b0_arr))

        He4_mass_frac_interp = 4 * Y_He4_interp

        # Plotting
        fig, axs = plt.subplots(
            3,
            1,
            figsize=figsize,
            sharex=True,
            height_ratios=[1, 3, 1],
        )
        # Plot 4Y_He4
        axs[0].plot(
            Omega_b0_arr,
            He4_mass_frac_interp,
            label="4Y_He4",
            color=BBN.COLORS[5],
        )

        # Plot errorbar area for observed value of 4Y_He4
        opacity = 0.3
        axs[0].fill_between(
            Omega_b0_arr,
            BBN.HE4_MASS_FRAC - BBN.HE4_MASS_FRAC_ERR,
            BBN.HE4_MASS_FRAC + BBN.HE4_MASS_FRAC_ERR,
            alpha=opacity,
            color=BBN.COLORS[5],
        )

        y_min = 0.2  # minimum value for y-axis in the top 4Y_He4 relic abundance plot
        y_max = 0.3  # max value -||-
        axs[0].set_ylim(bottom=y_min, top=y_max)
        axs[0].set_ylabel(r"$4Y_{He4}$")
        axs[0].tick_params(
            axis="both", which="both", direction="in", top=True, right=True
        )
        axs[0].legend()
        axs[0].set_xscale("log")
        axs[0].grid(True)

        # Plot Y_i/Y_p for D, He3, and Li7
        axs[1].loglog(
            Omega_b0_arr, Y_D_interp / Y_p_interp, label="D", color=BBN.COLORS[2]
        )
        axs[1].loglog(
            Omega_b0_arr,
            Y_He3_interp / Y_p_interp,
            label="He3",
            color=BBN.COLORS[4],
        )
        axs[1].loglog(
            Omega_b0_arr,
            Y_Li7_interp / Y_p_interp,
            label="Li7",
            color=BBN.COLORS[-2],
        )

        # Plot errorbar areas for observed values of Y_i/Y_p for D and Li7 (no error for He3)
        axs[1].fill_between(
            Omega_b0_arr,
            BBN.D_ABUNDANCE - BBN.D_ABUNDANCE_ERR,
            BBN.D_ABUNDANCE + BBN.D_ABUNDANCE_ERR,
            alpha=opacity,
            color=BBN.COLORS[2],
        )
        axs[1].fill_between(
            Omega_b0_arr,
            BBN.LI7_ABUNDANCE - BBN.LI7_ABUNDANCE_ERR,
            BBN.LI7_ABUNDANCE + BBN.LI7_ABUNDANCE_ERR,
            alpha=opacity,
            color=BBN.COLORS[-2],
        )

        y_min = 0.2e-10  # minimum value for y-axis in the middle Y_i/Y_p relic abundance plot
        axs[1].set_ylim(bottom=y_min)
        axs[1].set_ylabel(r"$Y_i/Y_p$")
        axs[1].tick_params(
            axis="both", which="both", direction="in", top=True, right=True
        )
        axs[1].legend()
        axs[1].grid(True)

        # Calculate likelihood as a function of Omega_b0, by using interpolated solutions
        likelihood, chi_sqr = np.asarray(
            [
                bayesian_probability(
                    np.asarray(
                        [
                            Y_D_interp[i],
                            Y_Li7_interp[i],
                            He4_mass_frac_interp[i],
                        ]
                    ),
                    np.asarray([BBN.D_ABUNDANCE, BBN.LI7_ABUNDANCE, BBN.HE4_MASS_FRAC]),
                    np.asarray(
                        [
                            BBN.D_ABUNDANCE_ERR,
                            BBN.LI7_ABUNDANCE_ERR,
                            BBN.HE4_MASS_FRAC_ERR,
                        ]
                    ),
                )
                for i in range(n_plot)
            ]
        ).T

        # Plot Bayesian likelihood
        axs[2].plot(Omega_b0_arr, likelihood, color="black")
        axs[2].set_xscale("log")
        axs[2].tick_params(
            axis="both", which="both", direction="in", top=True, right=True
        )
        axs[2].grid(True)
        axs[2].set_ylabel("Bayesian\nlikelihood")

        # Find the most likely Omega_b0 value by the min chi_squared value
        indx = np.argmin(chi_sqr)
        most_likely_chi_squared = np.min(chi_sqr)
        most_likely_Omega_b0 = Omega_b0_arr[indx]

        # Plot this as a dotted line on all three subplots
        for ax in axs:
            ax.axvline(
                most_likely_Omega_b0,
                color="black",
                linestyle="dotted",
                label=rf"$\chi^2$ = {most_likely_chi_squared:.3f}"
                + "\n"
                + rf"$\Omega_{{b0}}$ = {most_likely_Omega_b0:.5f}",
            )
        axs[2].legend()

        # Plot config
        fig.suptitle("Relic abundance analysis")
        fig.supxlabel(r"$\Omega_{b0}$")
        if filename:
            # Create fig dir directory if it doesn't already exist
            if not FIG_DIR.exists():
                FIG_DIR.mkdir()
            plt.savefig(filename)
        else:
            plt.show()

    @staticmethod
    def interpolate_Y_of_Neff(
        T_i: float,
        T_f: float,
        N_eff_vals: np.ndarray,
        Y_min: float = 1e-20,
    ) -> tuple[callable, callable, callable, callable]:
        """Static method. Interpolates the mass fractions Y_i as a function of N_eff to give a smooth
        function graph.

        arguments:
            T_i: initial temperature [K]
            T_f: final temperature [K]
            N_eff_vals: array of N_eff values to interpolate
            Y_min: lower bound value for mass fractions, everything below this value is set to Y_min

        returns:
            Tuple containing the interpolated mass fractions functions Y_i(N_eff)
            for i = p, D, He3, He4, and Li7
        """
        Y_model = np.zeros((5, len(N_eff_vals)))
        for i, N_eff in enumerate(N_eff_vals):
            # Initialize instance
            bbn = BBN(N_species=8, N_eff=N_eff)

            # Solve ode for this Omega_b0
            _, Y = bbn.solve_BBN(T_i, T_f)

            # Extract mass fraction values for final temperature
            Y = Y[:, -1]

            # T and Be7 decays to respectively He3 and Li7
            Y[4] += Y[3]
            Y[6] += Y[7]

            # Set lower bound for mass fractions
            Y[Y < Y_min] = Y_min

            # Extract mass fractions
            Y_p = Y[1]
            Y_D = Y[2]
            Y_He3 = Y[4]
            Y_He4 = Y[5]
            Y_Li7 = Y[6]
            Y = np.asarray([Y_p, Y_D, Y_He3, Y_He4, Y_Li7])

            # Append processed mass fractions
            Y_model[:, i] = Y

        # Interpolate each function in logspace
        kind = "cubic"
        Y_p, Y_D, Y_He3, Y_He4, Y_Li7 = Y_model

        Y_p_interp = interp1d(N_eff_vals, Y_p, kind=kind)
        Y_D_interp = interp1d(N_eff_vals, Y_D, kind=kind)
        Y_He3_interp = interp1d(N_eff_vals, Y_He3, kind=kind)
        Y_He4_interp = interp1d(N_eff_vals, Y_He4, kind=kind)
        Y_Li7_interp = interp1d(N_eff_vals, Y_Li7, kind=kind)

        return Y_p_interp, Y_D_interp, Y_He3_interp, Y_He4_interp, Y_Li7_interp

    @staticmethod
    def plot_relic_abundances_Neff(
        T_i: float,
        T_f: float,
        N_eff_vals: np.ndarray,
        Y_min: float = 1e-20,
        n_plot: int = 300,
        filename: str = None,
        figsize: tuple[int, int] = (7, 6),
    ) -> None:
        """Static method. Plots the relic abundances of elements Y_i/Y_p in the BBN process, as a function
        of the the number of neutrino species N_eff, by interpolating. Also finds the most likely
        N_eff value by using Bayesian probability.

        arguments:
            T_i: initial temperature [K]
            T_f: final temperature [K]
            N_eff_vals: array of N_eff values used to calculate before interpolation
            Y_min: lower bound value for mass fractions, everything below is set to this
            n_plot: number of points to use for plotting, after the interpolation
            filename: the filename to save the plot figure, if none the figure is shown
            figsize: the plot figure size
        """
        # Solve ODE for the values and interpolate
        (
            Y_p_interp_func,
            Y_D_interp_func,
            Y_He3_interp_func,
            Y_He4_interp_func,
            Y_Li7_interp_func,
        ) = BBN.interpolate_Y_of_Neff(T_i, T_f, N_eff_vals, Y_min=Y_min)

        # Array to interpolate graph with
        N_eff_arr = np.linspace(N_eff_vals[0], N_eff_vals[-1], n_plot)

        # Interpolate with these values
        Y_p_interp = Y_p_interp_func(N_eff_arr)
        Y_D_interp = Y_D_interp_func(N_eff_arr)
        Y_He3_interp = Y_He3_interp_func(N_eff_arr)
        Y_He4_interp = Y_He4_interp_func(N_eff_arr)
        Y_Li7_interp = Y_Li7_interp_func(N_eff_arr)

        He4_mass_frac_interp = 4 * Y_He4_interp

        # Plotting
        fig, axs = plt.subplots(
            4,
            1,
            figsize=figsize,
            sharex=True,
        )
        # Plot 4Y_He4
        axs[0].plot(
            N_eff_arr,
            He4_mass_frac_interp,
            label="4Y_He4",
            color=BBN.COLORS[5],
        )

        # Plot errorbar area for observed value of 4Y_He4
        opacity = 0.3
        axs[0].fill_between(
            N_eff_arr,
            BBN.HE4_MASS_FRAC - BBN.HE4_MASS_FRAC_ERR,
            BBN.HE4_MASS_FRAC + BBN.HE4_MASS_FRAC_ERR,
            alpha=opacity,
            color=BBN.COLORS[5],
        )

        y_min = 0.2  # minimum value for y-axis in the top 4Y_He4 relic abundance plot
        y_max = 0.3  # max value -||-
        axs[0].set_ylim(bottom=y_min, top=y_max)
        axs[0].set_ylabel(r"$4Y_{He4}$")
        axs[0].tick_params(
            axis="both", which="both", direction="in", top=True, right=True
        )
        axs[0].legend()
        axs[0].grid(True)

        # Plot Y_i/Y_p for D, He3
        axs[1].plot(N_eff_arr, Y_D_interp / Y_p_interp, label="D", color=BBN.COLORS[2])
        axs[1].plot(
            N_eff_arr,
            Y_He3_interp / Y_p_interp,
            label="He3",
            color=BBN.COLORS[4],
        )

        # Plot errorbar areas for observed values of Y_i/Y_p for D (no  error for He3)
        axs[1].fill_between(
            N_eff_arr,
            BBN.D_ABUNDANCE - BBN.D_ABUNDANCE_ERR,
            BBN.D_ABUNDANCE + BBN.D_ABUNDANCE_ERR,
            alpha=opacity,
            color=BBN.COLORS[2],
        )

        y_min = 1e-5
        y_max = 4e-5
        axs[1].set_ylim(bottom=y_min, top=y_max)
        axs[1].set_ylabel(r"$Y_i/Y_p$")
        axs[1].tick_params(
            axis="both", which="both", direction="in", top=True, right=True
        )
        axs[1].legend()
        axs[1].grid(True)

        # Plot Y_i/Y_p for Li7
        axs[2].plot(
            N_eff_arr,
            Y_Li7_interp / Y_p_interp,
            label="Li7",
            color=BBN.COLORS[-2],
        )

        # Plot errorbar areas for observed values of Y_i/Y_p for Li7
        axs[2].fill_between(
            N_eff_arr,
            BBN.LI7_ABUNDANCE - BBN.LI7_ABUNDANCE_ERR,
            BBN.LI7_ABUNDANCE + BBN.LI7_ABUNDANCE_ERR,
            alpha=opacity,
            color=BBN.COLORS[-2],
        )

        y_min = 1e-10
        y_max = 5e-10
        axs[2].set_ylim(bottom=y_min, top=y_max)
        axs[2].set_ylabel(r"$Y_i/Y_p$")
        axs[2].tick_params(
            axis="both", which="both", direction="in", top=True, right=True
        )
        axs[2].legend()
        axs[2].grid(True)

        # Calculate likelihood as a function of N_eff, by using interpolated solutions
        likelihood, chi_sqr = np.asarray(
            [
                bayesian_probability(
                    np.asarray(
                        [
                            Y_D_interp[i],
                            Y_Li7_interp[i],
                            He4_mass_frac_interp[i],
                        ]
                    ),
                    np.asarray([BBN.D_ABUNDANCE, BBN.LI7_ABUNDANCE, BBN.HE4_MASS_FRAC]),
                    np.asarray(
                        [
                            BBN.D_ABUNDANCE_ERR,
                            BBN.LI7_ABUNDANCE_ERR,
                            BBN.HE4_MASS_FRAC_ERR,
                        ]
                    ),
                )
                for i in range(n_plot)
            ]
        ).T

        # Plot Bayesian likelihood
        axs[3].plot(N_eff_arr, likelihood, color="black")
        axs[3].tick_params(
            axis="both", which="both", direction="in", top=True, right=True
        )

        # Find the most likely N_eff value by the min chi_squared value
        indx = np.argmin(chi_sqr)
        most_likely_chi_squared = np.min(chi_sqr)
        most_likely_N_eff = N_eff_arr[indx]

        # Plot this as a dotted line on all three subplots
        for ax in axs:
            ax.axvline(
                most_likely_N_eff,
                color="black",
                linestyle="dotted",
                label=rf"$\chi^2$ = {most_likely_chi_squared:.3f}"
                + "\n"
                + rf"$N_eff$ = {most_likely_N_eff:.5f}",
            )

        axs[3].legend()
        axs[3].grid(True)
        axs[3].set_ylabel("Bayesian\nlikelihood")

        # Plot config
        fig.suptitle("Relic abundance analysis")
        fig.supxlabel(r"$N_eff$")
        if filename:
            # Create fig dir directory if it doesn't already exist
            if not FIG_DIR.exists():
                FIG_DIR.mkdir()
            plt.savefig(filename)
        else:
            plt.show()
