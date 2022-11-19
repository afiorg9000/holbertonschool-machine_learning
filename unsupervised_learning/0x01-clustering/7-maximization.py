#!/usr/bin/env python3
"""maximization step in the EM algorithm for a GMM:"""
import numpy as np


def maximization(X, g):
    """maximization step in the EM algorithm for a GMM:"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None
    if type(g) is not np.ndarray or g.ndim != 2:
    n, d = X.shape
    k, _ = g.shape
    pr = np.sum(g, axis=0)
    n_val = np.sum(pr)
    if n_val != n:
        return None, None, None
    S = np.zeros((k, d, d))
    pi = np.zeros((k,))
    m = np.matmul(g, X) / np.sum(g, axis=1).reshape(-1, 1)
    for i in range(k):
        X_mean = X - m[i]
        S[i] = np.matmul(np.multiply(g[i], X_mean.T), X_mean) / np.sum(g[i])
        pi[i] = np.sum(g[i]) / n
    return pi, m, S
