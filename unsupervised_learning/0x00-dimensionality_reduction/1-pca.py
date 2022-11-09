!/usr/bin/env python3
"""performs PCA on a dataset:"""
import numpy as np

def pca(X, var=0.95):
    """performs PCA on a dataset:"""
    # Calculate the covariance matrix of X and find its eigenvalues and eigenvectors.
    cov_matrix = np.cov(X, rowvar=False)  # (n x d) * (d x d)T = (n x n)

    eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # Eigenvalues are sorted in descending order by default

    # Sort the components according to their explained variance: The first k columns of U represent the principal components in decreasing order of variance explained by each column of U . For example if you sort from largest to smallest explaination then the first column is most important for explaining 99% of the variance while the second column explains 87% and so on...
    idx = np.argsort(-eigvals)  # Descending order sort indices based on eigenvalue magnitude

    W = eigvecs[:, idx]  # Get top k weights that maintain 95% variance explained by data points

    return W