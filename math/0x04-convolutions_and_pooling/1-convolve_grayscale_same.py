#!/usr/bin/env python3
"""performs a same convolution on grayscale images:"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """performs a same convolution on grayscale images:"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph = kh // 2
    pw = kw // 2
    oh = h + 2 * ph - kh + int(kh % 2 == 1)
    ow = w + 2 * pw - kw + int(kw % 2 == 1)
    dim = (m, oh, ow)
    out = np.zeros(dim)
    padded = np.pad(images,
                    pad_width=((0, 0), (ph, ph),
                               (pw, pw)), mode='constant',
                    constant_values=0)

    for i in range(dim[1]):
        for j in range(dim[2]):
            x = i + kh
            y = j + kw
            M = padded[:, i:x, j:y]
            out[:, i, j] = np.tensordot(M, kernel)

    return out
