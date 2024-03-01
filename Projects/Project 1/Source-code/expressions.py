from typing import Callable
import numpy as np

# Constants
G = 6.6743e-11  # Newton gravitational constant [m^3/(kg s^2)]
alpha = 1
xi = 3 / 2


def gamma(V: str) -> float:
    """Describes Gamma(phi), equation 18 of the project, which depends on
    the potential

    arguments:
        V: the potential function in ["power", "exponential"]

    returns:
        the value of Gamma (eq. 18)
    """
    if V.lower() == "power":
        return (alpha + 1) / alpha
    elif V.lower() == "exponential":
        return 1
    else:
        raise ValueError("V-string value not recognized")


def dlambda(X: np.ndarray, V: str) -> float:
    """Describes dlambda/dN, equation 22 of the project, which depends on the potential.

    arguments:
        X: array with values of [x1, x2, x3, lmbda]
        V: the potential function: ["power", "exponential"]

    returns:
        the right hand side of dlambda/dN (eq. 22)
    """

    x1, _, _, lmbda = X

    return -np.sqrt(6) * lmbda**2 * (gamma(V) - 1) * x1


def ode_system(N: np.ndarray, X: np.ndarray, V: str) -> list[float]:
    """System of the four coupled ODE's from expression 19-22 of the project.

    arguments:
        N: characteristic time array (eq. 15)
        X: array with values of [x1, x2, x3, lmbda]
        V: the potential function in ["power", "exponential"]

    returns:
        array with the right hand sides of the ode's [dx1/dN, dx2/dN, dx3/dN, dlmbda/dN]
    """

    x1, x2, x3, lmbda = X

    hubble_expression = -0.5 * (3 + 3 * x1**2 - 3 * x2**2 + x3**2)
    dx1 = -3 * x1 + np.sqrt(6) / 2 * lmbda * x2**2 - x1 * hubble_expression
    dx2 = -np.sqrt(6) / 2 * lmbda * x1 * x2 - x2 * hubble_expression
    dx3 = -2 * x3 - x3 * hubble_expression
    dlmbda = dlambda(X, V)

    return [dx1, dx2, dx3, dlmbda]


def eos_integrand(omega_phi: float) -> float:
    """Represents the integrand over N of the equation of state (the projects equation 7)

    arguments:
        omega_phi: the eos parameter

    returns:
        the value at given input
    """
    return 3 * (1 + omega_phi)
