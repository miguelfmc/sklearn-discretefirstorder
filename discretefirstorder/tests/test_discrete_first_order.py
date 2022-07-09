"""
Test estimators
"""

import pytest
from sklearn.datasets import load_iris

from discretefirstorder import DFORegressor


@pytest.fixture
def data():
    """Load Iris dataset for testing"""
    return load_iris(return_X_y=True)


def test_dfo_regressor(data):
    """Test DFORegressor"""
    reg = DFORegressor()
    assert reg.loss == "mse"
    assert reg.learning_rate == "auto"
    assert reg.k == 3
    assert reg.polish is True
    assert reg.n_runs == 50
    assert reg.max_iter == 100
    assert reg.tol == 1e-3
    assert reg.fit_intercept is False
    assert reg.normalize is False

    reg.fit(*data)
    assert hasattr(reg, "is_fitted_")
