#!/usr/bin/env python3
"""initializes all variables required
to calculate the P affinities in t-SNE:"""
import numpy as np


def P_init(X, perplexity):
    """initializes all variables required to
    calculate the P affinities in t-SNE:"""
    n, d = X.shape
    D = np.zeros((n, n))
    X_sum = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), X_sum).T, X_sum)
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
