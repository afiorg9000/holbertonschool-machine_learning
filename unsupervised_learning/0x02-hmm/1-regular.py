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
    if P.shape[0] != P.shape[1]:
        return None

    n, _ = P.shape

    eigen_val, eigen_vec = np.linalg.eig(P.T)
    close_1 = np.isclose(eigen_val, 1)
    target_v = eigen_vec[:, close_1]
    target_v = target_v[:, 0]
    return (target_v / np.sum(target_v)).reshape(1, n)
