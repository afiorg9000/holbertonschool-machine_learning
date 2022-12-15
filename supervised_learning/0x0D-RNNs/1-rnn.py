#!/usr/bin/env python3
import numpy as np


class RNNCell:
    """class RNNCell"""
    def __init__(self, i, h, o):
        """performs forward propagation for a simple RNN"""
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """performs forward propagation for a simple RNN"""
        h_next = np.tanh(np.matmul(np.hstack((h_prev, x_t)),
                                   self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        return h_next, y

    def softmax(self, x):
        """performs forward propagation for a simple RNN"""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def sigmoid(self, x):
        """performs forward propagation for a simple RNN"""
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, X):
        """performs forward propagation for a simple RNN"""
        t = X.shape[0]
        m = X.shape[1]
        h = self.Wh.shape[1]
        H = np.zeros((t + 1, m, h))
        Y = np.zeros((t, m, self.Wy.shape[1]))
        for i in range(t):
            H[i + 1], Y[i] = self.forward(H[i], X[i])
        return H, Y


def rnn(rnn_cell, X, h_0):
    """performs forward propagation for a simple RNN"""
    t = X.shape[0]
    m = X.shape[1]
    h = h_0.shape[1]
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    H[0] = h_0
    for i in range(t):
        H[i + 1], Y[i] = rnn_cell.forward(H[i], X[i])
    return H, Y
