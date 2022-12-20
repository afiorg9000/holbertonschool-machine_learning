#!/usr/bin/env python3
"""tests for the optimum number of clusters by variance:"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """tests for the optimum number of clusters by variance:"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(kmin) is not int or kmin < 1:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax < 1:
        return None, None
    if kmax <= kmin:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None
    results = []
    vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append(C)
        results.append(clss)
        vars.append(variance(X, C))

    d0 = vars[0]
    d_vars = [d0 - v for v in vars]
    return results,
