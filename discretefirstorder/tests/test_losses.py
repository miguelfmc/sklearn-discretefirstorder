"""
Test losses
"""

from discretefirstorder._losses import MSELoss, BinaryCrossEntropyLoss


def test_mse_loss():
    """Test MSELoss"""
    import numpy as np

    y = np.array([2, 0, -1, 10])
    X = np.array([[1, 2], [-1, 0], [1, -1], [8, 3]])
    coef = np.array([1, 1])

    loss = MSELoss()
    loss_value = 2.0
    gradient_value = np.array([11, 4])

    assert loss.loss(coef, X, y) == loss_value
    assert np.array_equal(loss.gradient(coef, X, y), gradient_value)


def test_bce_loss():
    """Test BinaryCrossEntropyLoss"""
    import numpy as np

    y = np.array([1, 0, 0, 1])
    X = np.array([[1, 1], [-1, 0], [1, -1], [0, 1]])
    coef = np.array([1, 1])

    loss = BinaryCrossEntropyLoss()
    loss_value = 1.4465985666393635
    gradient_value = np.array([0.11185566, -0.88814434])

    assert loss.loss(coef, X, y) == loss_value
    assert np.array_equal(
        np.round(loss.gradient(coef, X, y), 4), np.round(gradient_value, 4)
    )
