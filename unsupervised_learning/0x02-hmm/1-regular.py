#!/usr/bin/env python3
"""determines the steady state probabilities"""
import numpy as np


def regular(P):
    """determines the steady state probabilities"""
    if type(P) is not np.ndarray or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if np.all(P * P) == 0:
        return None
    A = np.linalg.matrix_power(P, 100)
    return A[:, -1]
