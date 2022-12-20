#!/usr/bin/env python3
"""performs K-means on a dataset:"""
import numpy as np


def classes(X, C):
    """initializing the cluster"""
    Xe = np.expand_dims(X, axis=1)
    Ce = np.expand_dims(C, axis=0)
    D = np.sum(np.square(Xe - Ce), axis=2)
    clss = np.argmin(D, axis=1)
    return clss


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset:"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(k) is not int or int(k) != k or k < 1:
        return None, None
    if type(
        iterations) is not int or int(
            iterations) != iterations or iterations < 1:
        return None, None
    _, d = X.shape
    min = np.min(X, axis=0)
    max = np.max(X, axis=0)
    C = np.random.uniform(min, max, size=(k, d))
    nC = C.copy()
    for _ in range(iterations):
        clss = classes(X, C)
        for i in range(k):
            indices = np.argwhere(clss == i).reshape(-1)
            if X[indices].shape[0] > 0:
                nC[i] = np.mean(X[indices], axis=0)
            else:
                nC[i] = np.random.uniform(min, max)
        if np.array_equal(nC, C):
            break
        C = nC.copy()
    clss = classes(X, C)
    return C, clss
