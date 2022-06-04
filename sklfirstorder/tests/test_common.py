import pytest

from sklearn.utils.estimator_checks import check_estimator

from sklfirstorder import TemplateEstimator
from sklfirstorder import TemplateClassifier
from sklfirstorder import TemplateTransformer


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), TemplateTransformer(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
