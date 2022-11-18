#!/usr/bin/env python3
"""tests for the optimum number of clusters by variance:"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """tests for the optimum number of clusters by variance:"""
    results = []
    vars = []
    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)
        results.append(C)
        results.append(clss)
        vars.append(variance(X, C))

    d0 = vars[0]
    d_list = []
    for var in vars:
        d_list.append(d0 - var)
    return results, d_list
