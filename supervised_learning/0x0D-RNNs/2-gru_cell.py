#!/usr/bin/env python3
"""represents a gated recurrent unit:"""
import numpy as np


class GRUCell:
    """represents a gated recurrent unit:"""
    def __init__(self, i, h, o):
        """represents a gated recurrent unit:"""
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """represents a gated recurrent unit:"""
        xh = np.concatenate((h_prev, x_t), axis=1)
        z = np.matmul(xh, self.Wz) + self.bz
        r = np.matmul(xh, self.Wr) + self.br
        h_hat = np.matmul(xh, self.Wh) + self.bh
        h_hat = np.tanh(h_hat)
        h_next = (1 - z) * h_prev + z * h_hat
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y
