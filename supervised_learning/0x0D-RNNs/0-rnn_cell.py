#!/usr/bin/env python3
"""represents a gated recurrent unit:"""
import numpy as np


class RNNCell:
    """class RNNCell"""
    def __init__(self, i, h, o):
        """ represents a cell of a simple RNN:"""
        self.Wh = np.random.normal(size=(h+i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """represents a cell of a simple RNN:"""
        h_next = np.tanh(np.matmul(np.hstack((h_prev, x_t)),
                                   self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y
