#!/usr/bin/env python3
"""performs forward propagation for a bidirectional RNN:"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """performs forward propagation for a bidirectional RNN:"""
    t, m, i = X.shape
    h = h_0.shape[1]
    H = np.zeros((t + 1, m, 2 * h))
    H[0] = np.concatenate((h_0, h_t), axis=1)
    Y = np.zeros((t, m, i))
    for i in range(t):
        H[i + 1] = bi_cell.forward(H[i], X[i])
        Y[i] = np.matmul(H[i + 1], bi_cell.Wy) + bi_cell.by
    return H, Y
