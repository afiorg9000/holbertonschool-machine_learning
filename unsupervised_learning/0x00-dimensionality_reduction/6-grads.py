#!/usr/bin/env python3
"""calculates the gradients of Y:"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """Calculates the gradients of Y"""
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)
    dY = np.zeros((n, ndim))
    for i in range(n):
        dY[i] = np.sum(np.tile(P[:, i] + P[i, :], (ndim, 1)).T *
                       (Y[i] - Y), axis=0)
    return dY, Q
