"""
Test DFO algorithm and functions
"""

from discretefirstorder._dfo_optim import _threshold


def test_threshold():
    """Test _threshold"""
    import numpy as np

    arr = np.array([-10, 5, 1, 3, -4, 8, 2])
    out = np.array([-10, 5, 0, 0, 0, 8, 0])
    assert np.array_equal(_threshold(arr, 3), out)


def test_calculate_learning_rate():
    """Test _calculate_learning_rate"""
    pass


def test_solve_dfo():
    """Test _solve_dfo"""
    pass
