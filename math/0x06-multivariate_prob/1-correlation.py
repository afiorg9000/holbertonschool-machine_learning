#!/usr/bin/env python3
"""calculates a correlation matrix:"""
import numpy as np


def correlation(C):
    """calculates a correlation matrix:"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    d, _ = C.shape
    cor = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            cor[i][j] = C[i][j] / (np.sqrt(C[i][i] * C[j][j]))
    return cor
