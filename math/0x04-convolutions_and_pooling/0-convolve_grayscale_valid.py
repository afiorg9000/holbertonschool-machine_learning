#!/usr/bin/env python3
"""performs a valid convolution on grayscale images:"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """performs a valid convolution on grayscale images:"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    dim = (m, h - kh + 1, w - kw + 1)
    out = np.zeros(dim)
    for i in range(dim[1]):
        for j in range(dim[2]):
            x = i + kh
            y = j + kw
            M = images[:, i:x, j:y]
            out[:, i, j] = np.tensordot(M, kernel)
    return out
