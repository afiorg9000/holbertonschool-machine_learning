#!/usr/bin/env python3
"""calculates a correlation matrix:"""
import numpy as np


def correlation(C):
    """calculates a correlation matrix:"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]  # Dimension of the covariance matrix

    corr_matrix = (np.diag(1 / np.sqrt(np.diag(C)))).dot(
        (np.linalg.inv(C)).dot((np.diag(1 / np.sqrt(np.diag(C))))))

    return corr_matrix
