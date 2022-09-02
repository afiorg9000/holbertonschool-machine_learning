#!/usr/bin/env python3
"""normalizes an unactivated output using batch normalization:"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """normalizes an unactivated output using batch normalization:"""
    m = Z.shape[0]
    #  Compute the mean of the Zs in the layer
    mean = 1 / m * np.sum(Z, axis=0)
    #  compute the variance of the Zs
    variance = 1 / m * np.sum((Z - mean) ** 2, axis=0)
    #  Normalize the Z with mean and standard deviation
    Z_norm = (Z - mean) / (np.sqrt(variance + epsilon))
    #  Add some noise to the normalized Z so that they are reasonably different
    return (gamma * Z_norm) + beta
