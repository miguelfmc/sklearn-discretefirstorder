"""
Test estimators
"""

import pytest
from sklearn.datasets import load_iris, load_diabetes

from discretefirstorder import DFORegressor, _preprocess_data


@pytest.fixture
def iris_data():
    """Load Iris dataset for testing"""
    return load_iris(return_X_y=True)


@pytest.fixture
def diabetes_data():
    """Load diabetes dataset for testing"""
    return load_diabetes(return_X_y=True)


def test_dfo_regressor_default():
    """Test DFORegressor default params"""
    reg = DFORegressor()
    assert reg.loss == "mse"
    assert reg.learning_rate == "auto"
    assert reg.k == 1
    assert reg.polish is True
    assert reg.n_runs == 50
    assert reg.max_iter == 100
    assert reg.tol == 1e-3
    assert reg.fit_intercept is True
    assert reg.normalize is True


def test_dfo_regressor():
    """Test DFORegressor params"""
    reg = DFORegressor(
        learning_rate=0.001,
        k=5,
        polish=False,
        n_runs=25,
        max_iter=50,
        tol=1e-4,
        fit_intercept=False,
        normalize=False,
    )
    assert reg.loss == "mse"
    assert reg.learning_rate == 0.001
    assert reg.k == 5
    assert reg.polish is False
    assert reg.n_runs == 25
    assert reg.max_iter == 50
    assert reg.tol == 1e-4
    assert reg.fit_intercept is False
    assert reg.normalize is False


def test_dfo_regressor_fit(iris_data):
    """Test DFORegressor fit method"""
    reg = DFORegressor()
    reg.fit(*iris_data)
    assert hasattr(reg, "is_fitted_")


def test_invalid_loss():
    """Test invalid loss"""
    with pytest.raises(NotImplementedError):
        _ = DFORegressor(loss="myloss")


def test_invalid_value_learning_rate():
    """Test invalid string value for learning rate"""
    with pytest.raises(ValueError):
        _ = DFORegressor(learning_rate="myrate")


def test_invalid_k(iris_data):
    """Test invalid k given data"""
    with pytest.raises(ValueError):
        reg = DFORegressor(k=10)  # n_features = 4
        reg.fit(*iris_data)


def test_preprocess_data():
    """Test custom _preprocess_data function"""
    import numpy as np

    X = np.array([[2, 0.5, 120], [0, -0.3, 200], [7, 0.1, 40]])
    y = np.array([10, 12, -2])

    X_, y_, X_offset, y_offset, X_scale = _preprocess_data(
        X, y, fit_intercept=True, normalize=True
    )

    X_true = np.array(
        [
            [-3.39683110e-01, 1.22474487e00, 0.00000000e00],
            [-1.01904933e00, -1.22474487e00, 1.22474487e00],
            [1.35873244e00, -4.24918736e-17, -1.22474487e00],
        ]
    )
    y_true = np.array([3.33333333, 5.33333333, -8.66666667])

    X_offset_true = np.array([3.0e00, 1.0e-01, 1.2e02])
    y_offset_true = 6.666666666666667
    X_scale_true = np.array([2.94392029, 0.32659863, 65.31972647])

    assert X.shape == X_.shape
    assert y.shape == y_.shape

    assert np.array_equal(np.round(X_, 6), np.round(X_true, 6))
    assert np.array_equal(np.round(y_, 6), np.round(y_true, 6))

    assert np.array_equal(np.round(X_offset, 6), np.round(X_offset_true, 6))
    assert np.array_equal(np.round(y_offset, 6), np.round(y_offset_true, 6))
    assert np.array_equal(np.round(X_scale, 6), np.round(X_scale_true, 6))


# TODO implement test
@pytest.mark.skip(reason="test not implemented")
def test_preprocess_data_constant():
    """Test custom _preprocess_data function"""
    import numpy as np

    X = np.array([[1, 0.5, 120], [1, -0.3, 200], [1, 0.1, 40]])
    y = np.array([10, 12, -2])

    X_, y_, X_offset, y_offset, X_scale = _preprocess_data(
        X, y, fit_intercept=True, normalize=True
    )
    pass
