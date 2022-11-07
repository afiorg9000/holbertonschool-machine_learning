#!/usr/bin/env python3
"""calculates the mean and covariance of a data set:"""
import numpy as np


def mean_cov(X):
    """Calculates the mean and covariance of a data set."""

    # Calculate the mean: mean
    mean = np.mean(X, axis=0)

    # Subtract the mean from X: centered_x
    centered_x = X - mean

    # Calculate the covariance matrix using np.cov: cov
    cov = np.cov(centered_x.T)

    return (mean, cov)
