#!/usr/bin/env python3
"""calculates the probability density function:"""
import numpy as np


def pdf(X, m, S):
    """calculates the probability density function:"""
    _, d = X.shape

    X_mean = X - m
    inv = np.linalg.inv(S)
    det = np.linalg.det(S)
    result = 1 / np.sqrt(((2 * np.pi) ** d) * det)

    exp = np.sum(X_mean * np.matmul(inv, X_mean.T).T, axis=1)
    pdf = result * np.exp(-0.5 * exp)
    pdf[pdf < 1e-300] = 1e-300
    return pdf
