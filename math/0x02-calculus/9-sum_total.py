#!/usr/bin/env python3
"""function that evaluates a sigma formula"""
import numpy as np


def summation_i_squared(n):
    """function that evaluates a sigma formula"""
    if type(n) is not int or n < 1:
        return None
    squared = np.arange(1, n + 1)
    sum = np.sum(squared ** 2)
    return sum
