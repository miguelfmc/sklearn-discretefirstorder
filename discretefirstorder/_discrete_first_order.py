"""
Discrete First-Order Method for Classification and Regression
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.linalg import lstsq

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
# is it OK to use internal function?
from sklearn.linear_model._base import _preprocess_data
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from ._dfo_optim import _solve_dfo, _threshold


# TODO consider inheriting from LinearModel
class BaseDFO(BaseEstimator, metaclass=ABCMeta):
    """Base class for Discrete First Order classification and regression.
    """

    def __init__(self,
                 loss,
                 learning_rate="auto",
                 k=5,
                 polish=True,
                 n_runs=50,
                 max_iter=100,
                 tol=1e-3,
                 fit_intercept=False,
                 normalize=False):
        super(BaseDFO, self).__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.k = k
        self.polish = polish
        self.n_runs = n_runs
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.normalize = normalize

    @abstractmethod
    def fit(self, X, y):
        """Fit model.
        """


class DFOClassifier(ClassifierMixin, BaseDFO):
    """Discrete first-order classifier.

    Parameters
    ----------

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

    def __init__(self,
                 loss="log_loss",
                 learning_rate="auto",
                 k=5,
                 polish=True,
                 n_runs=50,
                 max_iter=100,
                 tol=1e-3):
        super(DFOClassifier, self).__init__(loss=loss,
                                            learning_rate=learning_rate,
                                            k=k,
                                            polish=polish,
                                            n_runs=n_runs,
                                            max_iter=max_iter,
                                            tol=tol)

    def fit(self, X, y):
        """Implementation of the fit method for the discrete first-order classifier.

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
    """Discrete first-order regressor.

    Parameters
    ----------
    loss :

    learning_rate :

    k :

    polish :

    n_runs :

    max_iter :

    tol :

    fit_intercept :

    normalize :

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
                 tol=1e-3,
                 fit_intercept=False,
                 normalize=False):
        # TODO validate inputs e.g. learning rate and loss
        super(DFORegressor, self).__init__(loss=loss,
                                           learning_rate=learning_rate,
                                           k=k,
                                           polish=polish,
                                           n_runs=n_runs,
                                           max_iter=max_iter,
                                           tol=tol,
                                           fit_intercept=fit_intercept,
                                           normalize=normalize)

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Set intercept (adapted from sklearn LinearModel)
        """
        if self.fit_intercept:
            self.coef_ = np.divide(self.coef_, X_scale)
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
        else:
            self.intercept_ = 0.0

    def fit(self, X, y, coef_init=None):
        """Implementation of the fit method for the discrete first-order regressor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values.
        coef_init : (optional) array-like, shape (n_features,)
            Initial value of regression coefficients


        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        # TODO call _validate_data instead
        X, y = check_X_y(X, y)

        # preprocess data (center and scale) as in other linear models
        # by default we expect fit_intercept = False and normalize = False, therefore no preprocessing
        X, y, X_offset, y_offset, X_scale = _preprocess_data(X,
                                                             y,
                                                             self.fit_intercept,
                                                             self.normalize)

        # init coefficients
        if coef_init is None:
            if X.shape[0] > X.shape[1]:
                coef_init, _, _, _ = lstsq(X, y)
            else:
                coef_init = X.T @ y
            coef_init = _threshold(coef_init, self.k)

        # optimize
        objective = float("inf")
        coef = coef_init
        coef_temp = coef_init

        # TODO can we parallelize n_runs?
        for _ in range(self.n_runs):
            coef_temp, objective_temp = _solve_dfo(coef=coef_temp,
                                                   X=X,
                                                   y=y,
                                                   learning_rate=self.learning_rate,
                                                   k=self.k,
                                                   loss=self.loss,
                                                   polish=self.polish,
                                                   max_iter=self.max_iter,
                                                   tol=self.tol)
            if objective_temp < objective:
                coef = coef_temp
                objective = objective_temp

        # coefficients for scaled features
        self.coef_ = coef
        # TODO consider using LinearModel's _set_intercept
        # self.intercept_ = self.y_.mean() - self.coef_.T @ self.X_.mean(axis=0)
        # rescale coefficients and set intercept
        self._set_intercept(X_offset, y_offset, X_scale)

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
