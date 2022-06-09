"""
Discrete First-Order Method for Classification and Regression
"""

from abc import ABCMeta, abstractmethod
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from ._dfo_optim import _plain_dfo


class BaseDFO(BaseEstimator, metaclass=ABCMeta):
    """ Base class for Discrete First Order classification and regression.
    """

    def __init__(self, demo_param='demo_param'):
        self.demo_param = demo_param

    @abstractmethod
    def fit(self, X, y):
        """Fit model.
        """


class DFOClassifier(ClassifierMixin, BaseDFO):
    """ Discrete first-order classifier.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    Examples
    --------
    >>> from discretefirstorder import DFOClassifier
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = DFOClassifier()
    >>> estimator.fit(X, y)
    DFOClassifier()
    """

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        # TODO call _validate_data instead
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        # TODO implement 'fit' method for DFOCLassifier

        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The output corresponding to each input sample.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        pass


class DFORegressor(RegressorMixin, BaseDFO):
    """ Discrete first-order regressor.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.

    Examples
    --------
    >>> from discretefirstorder import DFORegressor
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.random.normal(size=(100, ))
    >>> estimator = DFORegressor()
    >>> estimator.fit(X, y)
    DFORegressor()
    """

    def __init__(self,
                 loss="mse",
                 learning_rate="auto",
                 k=5,
                 polish=True,
                 n_runs=50,
                 max_iter=100,
                 tol=1e-3):
        super(DFORegressor, self).__init__(loss=loss,
                                           learning_rate=learning_rate,
                                           k=k,
                                           polish=polish,
                                           n_runs=n_runs,
                                           max_iter=max_iter,
                                           tol=tol)

    def fit(self, X, y, coef_init=None):
        """Implementation of the fit method for the discrete first-order regressor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        # TODO call _validate_data instead
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y

        # TODO implement 'fit' for DFORegressor
        # checks

        # init coefficients
        if coef_init is None:
            if self.X_.shape[0] > self.X_shape[1]:
                # TODO fit least squares
            else:
                # TODO get X.T @ y
            # TODO threshold coefficient vector

        # optimize
        # TODO implement dfo algorithm
        objective = ...
        coef = ...
        coef_temp = ...
        # TODO can we parallelize n_runs?
        for _ in range(self.n_runs):
            coef_temp, objective_temp = _plain_dfo(coef=coef_temp,
                                                   X=self.X_,
                                                   y=self.y_,
                                                   learning_rate=self.learning_rate,
                                                   k=k,
                                                   loss=self.loss,
                                                   polish=self.polish,
                                                   max_iter=self.max_iter,
                                                   tol=self.tol)
            # TODO check best objective and coefs

        self.coef_ = coef

        return self

    def predict(self, X):
        """Implementation of a prediction for the discrete first-order regressor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The output corresponding to each input sample
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        pass
