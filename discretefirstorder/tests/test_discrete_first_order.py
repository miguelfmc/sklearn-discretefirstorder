import pytest
import numpy as np

from sklearn.datasets import load_iris
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from discretefirstorder import DFORegressor
from discretefirstorder import DFOClassifier


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_dfo_regressor(data):
    reg = DFORegressor()
    assert reg.demo_param == 'demo_param'
    assert ...

    reg.fit(*data)
    assert hasattr(reg, 'is_fitted_')


def test_dfo_classifier(data):
    X, y = data
    clf = DFOClassifier()
    assert clf

    clf.fit(X, y)
    assert hasattr(clf, 'classes_')

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)
