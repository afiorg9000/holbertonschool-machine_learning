#!/usr/bin/env python3
"""calculates the integral of a polynomial:"""
from numpy import corrcoef


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial:"""
    coefficients = []
    if (poly == 0):
        return None

    if len(poly) == 0:
        return coefficients.append(C)

    for i in range(len(poly)):
        if i == 0:
            coefficients.append(C)
            coefficients.append(poly[i])
        else:
            coefficients.append(poly[i] / (i + 1))
    return coefficients
