#!/usr/bin/env python3
"""shuffles the data points in two matrices the same way:"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles the data points in two matrices the same way:"""
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
