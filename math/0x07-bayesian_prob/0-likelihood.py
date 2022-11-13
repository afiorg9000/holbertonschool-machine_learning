#!/usr/bin/env python3
import numpy as np
"""Calculates probable side affects"""


def likelihood(x, n, P):
    """Calculates probable side affects"""
    if type(n) is not int or n < 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        raise ValueError("x must be an integer that is \
            greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) != np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    for i in P:
        if i < 0 or i > 1:
            raise ValueError("All values in P must be in the range [0, 1]")

    prob = (np.math.factorial(n) / np.math.factorial(
        (n-x))) * np.power((1-P),
                           (n-x)) * np.power(P, x)
    return prob
