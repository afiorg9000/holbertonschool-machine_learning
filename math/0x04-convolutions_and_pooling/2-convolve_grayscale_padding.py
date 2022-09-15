#!/usr/bin/env python3
"""performs a convolution on grayscale images with custom padding:"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """performs a convolution on grayscale images with custom padding:"""
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding[0], padding[1]
    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1
    dim = (m, oh, ow)
    out = np.zeros(dim)
    padded = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw)),
                           mode='constant',  constant_values=0)

    for i in range(dim[1]):
        for j in range(dim[2]):
            x = i + kh
            y = j + kw
            M = padded[:, i:x, j:y]
            out[:, i, j] = np.tensordot(M, kernel)

    return out
