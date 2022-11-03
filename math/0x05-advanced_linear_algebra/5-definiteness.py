#!/usr/bin/env python3
"""calculates the definiteness of a matrix:"""
import numpy as np


def definiteness(matrix):
    """calculates the definiteness of a matrix:"""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.array_equal(matrix.T, matrix):
        return None
    values, _ = np.linalg.eig(matrix)
    if all([value > 0 for value in values]):
        return ("Positive definite")
    if all([value >= 0 for value in values]):
        return ("Positive semi-definite")
    if all([value < 0 for value in values]):
        return ("Negative definite")
    if all([value <= 0 for value in values]):
        return ("Negative semi-definite")
    else:
        return ("Indefinite")
