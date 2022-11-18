#!/usr/bin/env python3
"""initializes variables for a Gaussian Mixture Model:"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """initializes variables for a Gaussian Mixture Model:"""
    C, clss = kmeans(X, k)
    pi = np.array([1 / k] * k)
    m = C
    S = np.array([np.eye(X.shape[1])] * k)
    return pi, m, S
