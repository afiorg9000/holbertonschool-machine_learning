#!/usr/bin/env python3
"""initializes variables for a Gaussian Mixture Model:"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """initializes variables for a Gaussian Mixture Model:"""
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None
    if k > X.shape[0]:
        return None, None, None
    C, clss = kmeans(X, k)
    pi = np.array([1 / k] * k)
    m = C
    S = np.array([np.eye(X.shape[1])] * k)
    return pi, m, S
