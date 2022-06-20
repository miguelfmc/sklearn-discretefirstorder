"""
Implementation of the discrete first-order method for subset selection
"""

import numpy as np
from ._losses import MSELoss, LogLoss


LOSSES = {"mse": MSELoss,
          "logloss": LogLoss}


def _threshold(arr, k):
    """Threshold array to keep top k largest elements

    Parameters
    -----------
    arr : array-like, input array

    k : int, number of nonzero elements to keep

    Returns
    -------
    result : "thresholded" array
    """
    idx = np.argpartition(np.abs(arr), -k)[-k:]
    result = np.zeros_like(arr)
    result[idx] = arr[idx]
    return result


def _calculate_learning_rate(X):
    """Calculate learning rate based on design matrix X

    Parameters
    ----------
    X : design matrix

    Returns
    -------
    lr : float, learning rate based on Lipschitz constant
    """
    L = np.real(np.max(np.linalg.eigvals(X.T @ X)))
    return 1 / L


def _solve_dfo(coef, X, y, learning_rate, k, loss_type, polish, max_iter, tol):
    """Discrete first-order optimization routine.

    Parameters
    ----------
    coef :

    X

    y

    learning_rate

    k

    loss_type

    polish

    max_iter

    tol

    Returns
    -------
    coef :

    loss_value :

    """
    # check learning rate type and assign value to lr
    if isinstance(learning_rate, str):
        if learning_rate == "auto":
            lr = _calculate_learning_rate(X)
        else:
            raise NotImplementedError("No implemented logic for learning_rate:",
                                      learning_rate)

    elif isinstance(learning_rate, float):
        lr = learning_rate
    else:
        raise TypeError("learning_rate must be a float or 'auto'")

    # get loss from type
    loss = LOSSES[loss_type]

    # initialize stuff
    loss_value = loss.loss(coef, X, y)

    # algorithm loop
    for n_iter in range(max_iter):
        prev_loss_value = loss_value

        coef = coef - lr * loss.gradient(coef, X, y)
        coef = _threshold(coef, k)

        loss_value = loss.loss(coef, X, y)
        if (prev_loss_value - loss_value) < tol:
            break

    # polish
    if polish:
        # TODO implement coefficient polishing
        pass

    return coef, loss_value
