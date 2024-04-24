"""Statistical functions used in the project to calculate model probabilities."""

import numpy as np


def chi_squared(
    predicted: np.ndarray, observed: np.ndarray, error: np.ndarray
) -> float:
    """Calculates the chi-squared value between the predicted and observed values.

    arguments:
        predicted: the predicted values
        observed: the observed values
        error: the error in the observed values

    returns:
        the chi-squared value
    """
    return np.sum(((predicted - observed) / error) ** 2)


def bayesian_probability(
    predicted: np.ndarray, observed: np.ndarray, error: np.ndarray
) -> float:
    """Calculates the Bayesian probability for an array with predicted values,
    compared to array with observed values and array with the observed errors.
    Equation (28) of the project.

    arguments:
        predicted: the predicted values
        observed: the observed values
        error: the error in the observed values

    returns:
        the Bayesian probability, and the xi squared values
    """
    xi_sqr = chi_squared(predicted, observed, error)
    return (1 / (np.sqrt(2 * np.prod(error * error))) * np.exp(-xi_sqr)), xi_sqr
