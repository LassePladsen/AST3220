import numpy as np
from scipy.integrate import simpson

from problem9 import N_i, N_f
from problem10 import hubble_parameter_quintessence, hubble_parameter_lambdacdm


def universe_age_quintessence(V: str) -> float:
    """
    Calculate the dimensionless age of the universe H_0t_0 for the quintessence model

    arguments:
    V: V: the quintessence potential function in ["power", "exponential"]

    returns:
        The dimensionless age of the universe

    """
    z, h = hubble_parameter_quintessence(V, N_i, N_f)
    N = np.log(1 / (1 + z))
    return simpson(1/h, N)


def universe_age_lambdacdm() -> float:
    """
    Calculate the dimensionless age of the universe H_0t_0 for the Lambda-CDM model

    arguments:
        none
    returns:
        The dimensionless age of the universe

    """
    z = hubble_parameter_quintessence("power", N_i, N_f)[0]
    h = hubble_parameter_lambdacdm(z)
    N = np.log(1 / (1 + z))
    return simpson(1/h, N)


def print_universe_ages() -> None:
    """Prints the dimensionless ages of the universe for two quintessence models
    and the Lambda-CDM model

    arguments:
        none

    returns:
        none
    """
    print("Dimensionless age of the universe for quintessence models:")
    for V in ["power", "exponential"]:
        print(f"{V}-potential: {universe_age_quintessence(V)}")
    print()
    print(
        f"Dimensionless age of the universe for Lambda-CDM: {universe_age_lambdacdm()}"
    )

if __name__ == "__main__":
    print_universe_ages()
