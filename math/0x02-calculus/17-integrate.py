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

    for i, j in enumerate(poly):
        result = j / (i + 1)
        if int(result) == result:
            result = int(result)
        else:
            result
        coefficients.append(result)
    return coefficients
