import numpy as np
from scipy.integrate import simpson

from problem10 import hubble_parameter_quintessence, hubble_parameter_lambdacdm


def characteristic_age_quintessence(V: str) -> float:
    """
    Calculate the characteristic age of the universe H_0t_0 for the quintessence model

    arguments
    V: V: the quintessence potential function in ["power", "exponential"]

    returns
        The characteristic age of the universe

    """
    z, h = hubble_parameter_quintessence(V)
    N = np.log(1 / (1 + z))
    return simpson(h, N)


def characteristic_age_lambdacdm() -> float:
    """
    Calculate the characteristic age of the universe H_0t_0 for the Lambda-CDM model

    arguments
        none
    returns
        The characteristic age of the universe

    """
    z = hubble_parameter_quintessence("power")[0]
    h = hubble_parameter_lambdacdm(z)
    N = np.log(1 / (1 + z))
    return simpson(h, N)


def print_characteristic_ages() -> None:
    """Prints the characteristic ages of the universe for two quintessence models
    and the Lambda-CDM model
    
    arguments
        none
    
    returns
        none
    """
    print("Characteristic age of the universe for quintessence models:")
    for V in ["power", "exponential"]:
        print(f"{V}-potential: {characteristic_age_quintessence(V):e}")
    print()
    print(
        f"Characteristic age of the universe for Lambda-CDM: {characteristic_age_lambdacdm():e}"
    )

if __name__ == "__main__":
    print_characteristic_ages()