#!/usr/bin/env python3
"""performs forward propagation for a deep RNN:"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """performs forward propagation for a deep RNN:"""
    t, m, i = X.shape
    layers = len(rnn_cells)
    _, _, h = h_0.shape
    H = np.zeros((t + 1, layers, m, h))
    H[0, :, :, :] = h_0
    Y = []

    for i in range(t):
        for j in range(layers):
            if j == 0:
                h_next, y = rnn_cells[j].forward(H[i, j, :, :], X[i, :, :])
            else:
                h_next, y = rnn_cells[j].forward(H[i, j, :, :], h_next)
            H[i + 1, j, :, :] = h_next
        Y.append(y)
    Y = np.array(Y)
    return H, Y
