import pytest

from sklearn.utils.estimator_checks import check_estimator

from discretefirstorder import TemplateEstimator
from discretefirstorder import TemplateClassifier
from discretefirstorder import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
