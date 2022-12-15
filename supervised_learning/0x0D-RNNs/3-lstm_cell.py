#!/usr/bin/env python3
"""represents an LSTM unit:"""
import numpy as np


class LSTMCell:
    """represents an LSTM unit:"""

    def __init__(self, i, h, o):
        """represents an LSTM unit:"""
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """represents an LSTM unit:"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """represents an LSTM unit:"""
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """represents an LSTM unit:"""
        h_x = np.concatenate((h_prev, x_t), axis=1)
        f = self.sigmoid(np.matmul(h_x, self.Wf) + self.bf)
        u = self.sigmoid(np.matmul(h_x, self.Wu) + self.bu)
        c_hat = np.tanh(np.matmul(h_x, self.Wc) + self.bc)
        c_next = f * c_prev + u * c_hat
        o = self.sigmoid(np.matmul(h_x, self.Wo) + self.bo)
        h_next = o * np.tanh(c_next)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, c_next, y
