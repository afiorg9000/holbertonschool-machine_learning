#!/usr/bin/env python3
"""that converts a numeric label vector into a one-hot matrix:"""
import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix:"""
    m = Y.shape[0]

    if len(Y.shape) != 1:
        return None
    out = np.zeros((classes, m))

    for i in range(0, m):
        out[Y[i]][i] = 1
    return out
