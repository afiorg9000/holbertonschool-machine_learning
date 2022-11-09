#!/usr/bin/env python3
"""performs PCA on a dataset:"""
import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset:"""
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    covar = np.dot(X_centered.T, X_centered) / len(X)

    eigvals, eigvecs = np.linalg.eigh(covar)

    idx = eigvals.argsort()[::-1]

    T = eigvecs[:, idx][:, :ndim]

    return T
