#!/usr/bin/env python3
"""conducts forward propagation using Dropout:"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout:"""
    cache = {}
    cache["A0"] = X
    for i in range(L):
        if i == 0:
            cache["A0"] = X
        else:
            Z = np.dot(weights["W" + str(i + 1)],
                       cache["A" + str(i - 1)]) + (weights["b" + str(i)])
            cache["A" + str(i)] = np.tanh(Z) / keep_prob
    return cache
