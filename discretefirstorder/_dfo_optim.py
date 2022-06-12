"""
Implementation of the discrete first-order method for subset selection
"""

import numpy as np


def _threshold(arr, k):
    """Threshold array to keep top k largest elements
    """
    idx = np.argpartition(np.abs(arr), -k)[-k:]
    result = np.zeros_like(arr)
    result[idx] = arr[idx]
    return result


def _solve_dfo():
    pass