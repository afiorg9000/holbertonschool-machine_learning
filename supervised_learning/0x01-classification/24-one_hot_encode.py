#!/usr/bin/env python3
import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix:"""
    Y = np.asarray(Y)
    if len(Y.shape) != 1:
        return None
    out = np.zeros((classes, Y.shape[0]))

    for i in range(0, Y.shape[0]):
        index = int(Y[i]) - 1
        out[index][i] = 1
    return out
