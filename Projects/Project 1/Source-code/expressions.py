from typing import Callable
import numpy as np
from autograd import grad

G = 6.6743e-11  # Newton gravitational constant [m^3/(kg s^2)]


def V_power_law(phi: float, M=float) -> float:
    """Expression 23 of the project

    arguments:
        phi: field parameter
        M: mass scale

    returns:
        the value of the potential
    """
    alpha = 1
    return M ** (4 + alpha) * phi ** (-alpha)


def V_exponential(phi: float, V0: float) -> float:
    """Expression 24 of the project.

    arguments:
        phi: field parameter
        V0: the potential constant

    returns:
        the value of the potential
    """
    kappa = np.sqrt(8 * np.pi * G)
    xi = 3 / 2
    return V0 * np.exp(-kappa * xi * phi)


def ode_system(X: np.ndarray, N: str, V: str) -> np.ndarray:
    """System of the three coupled ODE's from expression 19-22 of the project.

    arguments:
        X: array of values of [x1, x2, x3, lmbda]
        N: time variable (expression 15)
        V: the potential function: ["power", "exponential"]

    returns:
        array of the ode's [dx1/dN, dx2/dN, dx3/dN, dlmbda/dN]
    """

    x1, x2, x3, lmbda = X

    temp = 3 + 3 + x1**2 - 3 * x2**2 + x3**2
    dx1 = -3 * x1 * np.sqrt(6) / 2 * lmbda * x2**2 * 1 / 2 * x1 * temp
    dx2 = -np.sqrt(6) / 2 * lmbda * x1 * x2 + 1 / 2 * x2 * temp
    dx3 = -2 * x3 + 1 / 2 * x3 * temp
    dlmbda = dlambda(V)

    return [dx1, dx2, dx3, dlmbda]

def dlambda(V: str) -> float:
    """Describes equation 22 of the project, which depends on the potential.

    arguments:
        V: the potential function: ["power", "exponential"]
    
    returns:
        the right hand side of dlambda/dN (eq 22)
    """

    if V.lower() == "power":
        return ...
    elif V.lower() == "exponential":
        return ...
