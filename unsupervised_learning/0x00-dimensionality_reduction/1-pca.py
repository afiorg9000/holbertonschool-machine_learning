#!/usr/bin/env python3
"""performs PCA on a dataset:"""
import numpy as np


def pca(X, ndim):
    """performs PCA on a dataset:"""
    X_mean = X - np.mean(X, axis=0)
    u, s, vr = np.linalg.svd(X_mean)
    W = vr[:ndim].T
    T = np.matmul(X_mean, W)
    return T
