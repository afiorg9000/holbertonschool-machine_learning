#!/usr/bin/env python3
"""forward propagation over a convolutional layer"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """forward propagation over a convolutional layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = (((sh * h_prev) - sh + kh - h_prev) // 2)
        pw = (((sw * w_prev) - sw + kw - w_prev) // 2)
    if padding == 'valid':
        ph, pw = 0, 0

    A_prev = np.pad(A_prev, [(0, 0), (ph, ph), (pw, pw),
                             (0, 0)], 'constant', constant_values=0)
    zh = ((h_prev + (2 * ph) - kh) // sh) + 1
    zw = ((w_prev + (2 * pw) - kw) // sw) + 1
    convolution = np.zeros((m, zh, zw, c_new))
    i = np.arange(0, m)

    for h in range(zh):
        for w in range(zw):
            for z in range(c_new):
                convolution[i, h, w, z] = activation(np.sum(
                    np.multiply(A_prev[i, h * sh:kh + h * sh,
                                       w * sw:kw + w * sw],
                                W[:, :, :, z]),
                    axis=(1, 2, 3))) + b[0, 0, 0, z]

    return convolution
