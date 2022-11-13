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


def intersection(x, n, P, Pr):
    """calculates intersection with probabilities"""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    VE = "x must be an integer that is greater than or equal to 0"
    if type(x) is not int or x < 0:
        raise ValueError(VE)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    VE = "Pr must be a numpy.ndarray with the same shape as P"
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError(VE)

    if [x for x in P if x < 0 or x > 1]:
        raise ValueError("All values in P must be in the range [0, 1]")

    if [x for x in Pr if x < 0 or x > 1]:
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose([np.sum(Pr)], [1]):
        raise ValueError("Pr must sum to 1")

    P_p = likelihood(x, n, P) * Pr
    return P_p


def marginal(x, n, P, Pr):
    """calculates marginal probability"""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")

    VE = "x must be an integer that is greater than or equal to 0"
    if type(x) is not int or x < 0:
        raise ValueError(VE)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    VE = "Pr must be a numpy.ndarray with the same shape as P"
    if type(Pr) is not np.ndarray or Pr.shape != P.shape:
        raise TypeError(TE)

    if [x for x in P if x < 0 or x > 1]:
        raise ValueError("All values in P must be in the range [0, 1]")

    if [x for x in Pr if x < 0 or x > 1]:
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose([np.sum(Pr)], [1]):
        raise ValueError("Pr must sum to 1")

    P_p = intersection(x, n, P, Pr)
    return np.sum(P_p)
