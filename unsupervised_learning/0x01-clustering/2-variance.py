#!/usr/bin/env python3
"""calculates the total intra-cluster variance for a data set:"""
import numpy as np


def variance(X, C):
    """calculates the total intra-cluster variance for a data set:"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None
    if type(C) is not np.ndarray or C.ndim != 2 or C.shape[1] != X.shape[1]:
        return None
    N = np.expand_dims(X, axis=1)
    K = np.expand_dims(C, axis=0)
    dist = np.sum(np.square(N - K), axis=2)
    dist_C = np.min(dist, axis=1)
    var = np.sum(dist_C)
    return var
