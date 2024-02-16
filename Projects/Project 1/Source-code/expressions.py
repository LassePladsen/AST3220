import numpy as np

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
    return V0 * np.exp(-kappa*xi*phi)
