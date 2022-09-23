#!/usr/bin/env python3
"""performs forward propagation over a pooling layer of a neural network:"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """forward propagation over a pooling layer of a neural network:"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    zh = int(((h_prev - kh) // sh) + 1)
    zw = int(((w_prev - kw) // sw) + 1)
    convolution = np.zeros((m, zh, zw, c_prev))
    i = np.arange(m)

    for h in range(zh):
        for w in range(zw):
            if mode == 'max':
                convolution[i, h, w] = np.max(A_prev[i, h * sh:kh + (h * sh),
                                                     w * sw:kw + (w * sw)],
                                              axis=(1, 2))
            if mode == 'avg':
                convolution[i, h, w] = np.mean(A_prev[i, h * sh:kh + (h * sh),
                                                      w * sw:kw + (w * sw)],
                                               axis=(1, 2))
    return convolution
