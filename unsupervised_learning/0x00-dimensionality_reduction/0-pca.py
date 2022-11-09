#!/usr/bin/env python3
"""performs PCA on a dataset:"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset:"""
    cov_matrix = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(-eigvals)
    W = eigvecs[:, idx]
    return W
