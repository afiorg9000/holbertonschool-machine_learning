#!/usr/bin/env python3
"""that converts a numeric label vector into a one-hot matrix:"""
import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix:"""
    m = Y.shape[0]

    try:
        Y_one_hot = np.zeros((classes, m))
        Y_one_hot[Y, np.arange(m)] = 1
        return Y_one_hot
    except Exception:
        return None
