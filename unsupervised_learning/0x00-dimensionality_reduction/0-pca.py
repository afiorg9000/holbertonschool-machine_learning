#!/usr/bin/env python3
"""performs PCA on a dataset:"""
import numpy as np


def pca(X, var=0.95):
    """performs PCA on a dataset:"""
    cov_matrix = np.cov(X, rowvar=False)
    # (n x d) * (d x d)T = (n x n)

    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    # Eigenvalues are sorted in descending order by default

    idx = np.argsort(-eigvals)
    # Descending order sort indices based on eigenvalue magnitude

    W = eigvecs[:, idx]
    # Get top k weights that maintain 95% variance explained by data points

    return W
