#!/usr/bin/env python3
"""calculates the mean and covariance of a data set:"""
import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set."""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    # Calculate the mean: mean
    mean = np.mean(X, axis=0)
    mean = mean.reshape(1, d)
    # Subtract the mean from X: centered_x
    X_mean = X - mean
    # Calculate the covariance matrix using np.cov: cov
    cov = np.matmul(X_mean.T, X_mean) / (n - 1)
    return (mean, cov)
