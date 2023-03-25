"""
Implementation of some basic losses and their gradients
"""

import numpy as np
from scipy.special import expit


class MSELoss:
    """Implementation of squared loss (equivalent to MSE)
        and its gradient, for a linear model.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([2, 0, -1, 10])
    >>> X = np.array([[1, 2], [-1, 0], [1, -1], [8, 3]])
    >>> coef = np.array([1, 1])
    >>> loss = MSELoss()
    >>> loss.loss(coef, X, y)
    2.0
    >>> loss.gradient(coef, X, y)
    array([11,  4])
    """

    def __init__(self):
        pass

    @staticmethod
    def loss(coef, X, y):
        """Calculate squared loss for a linear model
            given the coefficients, data and target.

        Parameters
        ----------
        coef : array-like of shape (n_features,)
            coefficient vector.
        X : ndarray of shape (n_samples, n_features)
            training data.
        y : array-like of shape (n_samples,)
            target values.

        Returns
        -------
        loss_value : float
            loss value.
        """
        return 0.5 * np.sum((y - X @ coef) ** 2)

    @staticmethod
    def gradient(coef, X, y):
        """Calculate squared loss gradient for a linear model
            given the coefficients, data and target.

        Parameters
        ----------
        coef : array-like of shape (n_features,)
            coefficient vector.
        X : ndarray of shape (n_samples, n_features)
            training data.
        y : array-like of shape (n_samples,)
            target values.

        Returns
        -------
        gradient : ndarray of shape (n_features,)
            gradient of the squared loss w.r.t. coefficient vector.
        """
        return -X.T @ (y - X @ coef)


class BinaryCrossEntropyLoss:
    """Implementation of binary cross entropy loss
        and its gradient, for a linear model.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 0, 0, 1])
    >>> X = np.array([[1, 1], [-1, 0], [1, -1], [0, 1]])
    >>> coef = np.array([1, 1])
    >>> loss = BinaryCrossEntropyLoss()
    >>> loss.loss(coef, X, y)
    1.4465985666393635
    >>> loss.gradient(coef, X, y)
    array([ 0.11185566, -0.88814434])
    """

    def __init__(self):
        pass

    @staticmethod
    def loss(coef, X, y):
        """Calculate binary cross entropy loss for a linear model
            given the coefficients, data and target.

        Parameters
        ----------
        coef : array-like of shape (n_features,)
            coefficient vector.
        X : ndarray of shape (n_samples, n_features)
            training data.
        y : array-like of shape (n_samples,)
            target values.

        Returns
        -------
        loss_value : float
            loss value.
        """
        y_pred = expit(X @ coef)
        return -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    @staticmethod
    def gradient(coef, X, y):
        """Calculate binary cross entropy loss gradient for a linear model
            given the coefficients, data and target.

        Parameters
        ----------
        coef : array-like of shape (n_features,)
            coefficient vector.
        X : ndarray of shape (n_samples, n_features)
            training data.
        y : array-like of shape (n_samples,)
            target values.

        Returns
        -------
        gradient : ndarray of shape (n_features,)
            gradient of the binary cross entropy loss w.r.t. coefficient vector.
        """
        y_pred = expit(X @ coef)
        return X.T @ (y_pred - y)
