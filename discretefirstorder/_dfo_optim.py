"""
Implementation of the discrete first-order method for subset selection
"""

import numpy as np
from scipy.linalg import lstsq
from ._losses import MSELoss, LogLoss


LOSSES = {"mse": MSELoss,
          "logloss": LogLoss}


def _threshold(arr, k):
    """Threshold array to keep top k largest elements (in absolute value)

    Parameters
    -----------
    arr : array-like of shape (n_features,)
        input array.

    k : int
        number of nonzero elements to keep.

    Returns
    -------
    result : ndarray of shape (n_features,)
        'thresholded' array

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([-10, 5, 1, 3, -4, 8, 2])
    >>> _threshold(a, 3)
    array([-10,  5,  0,  0,  0,  8,  0])
    """
    idx = np.argpartition(np.abs(arr), -k)[-k:]
    result = np.zeros_like(arr)
    result[idx] = arr[idx]
    return result


def _calculate_learning_rate(X):
    """Calculate learning rate based on data X

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        training data.

    Returns
    -------
    lr : float
        learning rate based on Lipschitz constant.
    """
    L = np.real(np.max(np.linalg.eigvals(X.T @ X)))
    return 1 / L


def _solve_dfo(coef, X, y, learning_rate, k, loss_type, polish, max_iter, tol):
    """Discrete first-order optimization routine.

    Parameters
    ----------
    coef : array-like of shape (n_features,)
        coefficient vector.

    X : ndarray of shape (n_samples, n_features)
        training data.

    y : array-like of shape (n_samples,)
        target values.

    learning_rate : float
        learning rate.

    k : int
        number of non-zero coefficients to keep.

    loss_type : string
        loss type (one of 'mse' or 'logloss')

    polish : bool
        whether to polish coefficient vector by computing the least squares solution on active set

    max_iter : int
        max number of iterations

    tol : float
        tolerance.

    Returns
    -------
    coef : ndarray of shape (n_features,)
        coefficient vector.

    loss_value : float
        loss value.
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
    # TODO test polishing
    if polish:
        active_idx = (np.abs(coef) > 0)
        X_ = X[:, active_idx]

        polished, _, _, _ = lstsq(X_, y)
        coef[active_idx] = polished

        loss_value = loss.loss(coef, X, y)

    return coef, loss_value
