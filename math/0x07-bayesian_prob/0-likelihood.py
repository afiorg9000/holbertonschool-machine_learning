#!/usr/bin/env python3
"""Calculates probable side affects"""
import numpy as np


def likelihood(x, n, P):
    """Calculates probable side affects"""
    if type(n) is not int or n < 0:
        raise ValueError("n must be a positive integer")

    if type(x) is not int or x < 0:
        VE = "x must be an integer that is greater than or equal to 0"
        raise ValueError(VE)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) != np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    for i in P:
        if i < 0 or i > 1:
            raise ValueError("All values in P must be in the range [0, 1]")

    P_p = np.zeros(P.shape)
    n_f = np.math.factorial(n)
    x_f = np.math.factorial(x)
    p_f = np.math.factorial(n - x)
    for i in range(len(P)):
        P_p[i] = (n_f / (p_f * x_f)) * (P[i] ** x) * ((1 - P[i]) ** (n - x))
    return P_p
