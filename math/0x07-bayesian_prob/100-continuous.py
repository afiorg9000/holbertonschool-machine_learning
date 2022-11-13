#!/usr/bin/env python3
"""calculates posterior probability"""
from scipy import special


def posterior(x, n, p1, p2):
    """calculates posterior probability"""
    if (type(n) is not int) or (n <= 0):
        raise ValueError("n must be a positive integer")

    if (type(x) is not int) or (x < 0):
        VE = "x must be an integer that is greater than or equal to 0"
        raise ValueError(VE)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if type(p1) is not float or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")

    if type(p2) is not float or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")

    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    b2 = special.btdtr(x + 1, n - x + 1, p2)
    b1 = special.btdtr(x + 1, n - x + 1, p1)
    p = b2 - b1

    return p
