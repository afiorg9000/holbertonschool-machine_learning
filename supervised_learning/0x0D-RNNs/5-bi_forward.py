#!/usr/bin/env python3
"""represents a bidirectional cell of an RNN:"""
import numpy as np


class BidirectionalCell:
    """represents a bidirectional cell of an RNN:"""
    def __init__(self, i, h, o):
        """represents a bidirectional cell of an RNN:"""
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """represents a bidirectional cell of an RNN:"""
        h_next = np.tanh(np.matmul(np.hstack((h_prev, x_t)),
                                   self.Whf) + self.bhf)
        return h_next
