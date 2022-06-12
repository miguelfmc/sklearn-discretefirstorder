import pytest

from sklearn.utils.estimator_checks import check_estimator

from discretefirstorder import DFOClassifier
from discretefirstorder import DFORegressor


@pytest.mark.parametrize(
    "estimator",
    [DFORegressor(), DFOClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
