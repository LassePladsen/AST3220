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
FIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "figures")
)

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

        # Initialize the reaction rates and background
        self.RR = ReactionRates()
        self.background = Background(**background_kwargs)

        # Create this array if it doesn't already exist
        if not os.path.exists(FIG_DIR):
            os.makedirs(FIG_DIR)

        # Initialize empty variables
        self.Y = None
        self.T = None

    def _get_ode_system(self, lnT: np.ndarray, Y: np.ndarray) -> list[float]:
        """The right hand side of the coupled ODE system, for Y_i ODE's.

        arguments:
            lnT: logarithmic temperature array
            Y: array with initial values of Y_i like so [Y_n, Y_p, ...]

        returns:
            The right hand sides of the ode's
        """

        dY = np.zeros_like(Y)
        T = np.exp(lnT)
        T_9 = T * 1e-9

        # Neutrons and protons are always included
        Y_n = Y[0]
        Y_p = Y[1]

        lambda_np, lambda_pn = self.RR.get_weak_rates(T_9)

        # Change for left hand side of the ODE system (n <-> p)
        LHS_change = Y_p * lambda_pn - Y_n * lambda_np

        # Update neutron and proton ODE's
        dY[0] += LHS_change
        dY[1] -= LHS_change

        if self.N_species > 2:  # Include deuterium
            Y_d = Y[2]  

            # n+p <-> D + gamma 
            Y_np = Y_n * Y_p

            



        if self.N_species > 3:  # include trituim
            ...

        if self.N_species > 4:  # include helium-3
            ...

        if self.N_species > 5:  # include helium-4
            ...

        if self.N_species > 6:  # include lithium-7
            ...

        if self.N_species > 7:  # include beryllium-7
            ...

        return -dY / self.background.H(T)

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

    def plot_relative_number_densities(
        self,
        filename: str,
        figsize: tuple[int, int] = (7, 5),
        ymin: float = 1e-3,
        ymax: float = 2.0,
    ) -> None:
        """Plots the relative number densities for each species, as a function of logarithmic temperature ln(T)

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

        total = 0  # total sum of relative number densities
        for i, y in enumerate(self.Y):
            # Plot relative number density of species i
            ax.loglog(self.T, y, label=self.species_labels[i])

            # Add to total sum
            total += y

        # Plot thermal equilibrium value of neutron and proton
        ax.loglog(self.T, self._Y_n_equil(self.T), color="C0", linestyle=":")
        ax.loglog(self.T, self._Y_p_equil(self.T), color="C1", linestyle=":")

        # Finally plot the sum of the relative densities, which should always be equal to one
        ax.loglog(self.T, total, "k:", label="Sum")

        # Plot settings
        plt.gca().invert_xaxis()  # invert x-axis
        plt.xlabel("T [K]")
        plt.ylabel(r"$Y_i$")

        plt.ylim(bottom=ymin, top=ymax)
        plt.legend()
        plt.grid()
        plt.title("Relative number densities of particles species")
        plt.savefig(filename)


if __name__ == "__main__":
    ### DIRECT USAGE EXAMPLE ###

    # Variables
    N_species = 2  # number of interacting atom species
    N_eff = 3  # effective number of neutrino species
    T_i = 1e11  # initial temperature [K]
    T_f = 1e8  # final temperature [K]
    n_points = 1000  # number of points for plotting
    unit = "cgs"  # unit system to use

    # Initialize
    bbn = BBN(N_species, N_eff=N_eff, unit=unit)

    # Solve ode
    bbn.solve_ode_system(T_i, T_f, n_points)

    # Plot
    filename = os.path.join(FIG_DIR, "example_problem_f.png")
    bbn.plot_relative_number_densities(filename)
