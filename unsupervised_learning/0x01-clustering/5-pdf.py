#!/usr/bin/env python3
"""calculates the probability density function:"""
import numpy as np


def pdf(X, m, S):
    """calculates the probability density function:"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None
    if type(S) is not np.ndarray or S.ndim != 2 or S.shape[0] != S.shape[1]:
        return None
    if S.shape[0] != m.shape[0]:
        return None
    _, d = X.shape

    X_mean = X - m
    inv = np.linalg.inv(S)
    det = np.linalg.det(S)
    result = 1 / np.sqrt(((2 * np.pi) ** d) * det)

    exp = np.sum(X_mean * np.matmul(inv, X_mean.T).T, axis=1)
    pdf = result * np.exp(-0.5 * exp)
    pdf[pdf < 1e-300] = 1e-300
    return pdf
