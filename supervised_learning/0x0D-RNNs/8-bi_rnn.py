#!/usr/bin/env python3
"""performs forward propagation for a bidirectional RNN:"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """performs forward propagation for a bidirectional RNN:"""
    t, m, i = X.shape
    h, _ = h_0.shape
    H = np.zeros((t, m, 2 * h))
    H_f = np.zeros((t, m, h))
    H_b = np.zeros((t, m, h))
    H_f[0] = h_0
    H_b[t - 1] = h_t
    Y = np.zeros((t, m, bi_cell.Whf.shape[1]))
    for i in range(t):
        if i > 0:
            H_f[i] = bi_cell.forward(H_f[i - 1], X[i])
        if i < t - 1:
            H_b[t - i - 2] = bi_cell.backward(H_b[t - i - 1], X[t - i - 1])
        H[i] = np.concatenate((H_f[i], H_b[i]), axis=1)
        Y[i] = bi_cell.output(H[i])
    return H, Y