#!/usr/bin/env python3
"""performs PCA on a dataset:"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset:"""
    u, s, vr = np.linalg.svd(X)
    k = np.sum(np.cumsum(s) / np.sum(s) < var)
    W = vr.T[:, :k + 1]
    return W
