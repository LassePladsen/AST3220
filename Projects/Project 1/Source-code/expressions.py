from typing import Callable
import numpy as np

# Constants
G = 6.6743e-11  # Newton gravitational constant [m^3/(kg s^2)]
alpha = 1
xi = 3 / 2


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
    return V0 * np.exp(-kappa * xi * phi)


def gamma(V: str) -> float:
    """Describes Gamma(phi), equation 18 of the project, which depends on
    the potential

    arguments:
        V: the potential function in ["power", "exponential"]

    returns:
        the value of Gamma (eq. 18)
    """
    if V.lower() == "power":
        return 1 / alpha
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

    x1, x2, x3, lmbda = X

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
    # print(x1)
    # print(x1,x2,x3,lmbda)

    temp = 3 + 3 + x1**2 - 3 * x2**2 + x3**2
    # print(temp)
    dx1 = -3 * x1 + np.sqrt(6) / 2 * lmbda * x2**2 + 1 / 2 * x1 * temp
    # print(dx1)
    dx2 = -np.sqrt(6) / 2 * lmbda * x1 * x2 + 1 / 2 * x2 * temp
    dx3 = -2 * x3 + 1 / 2 * x3 * temp
    dlmbda = dlambda(X, V)

    return [dx1, dx2, dx3, dlmbda]
