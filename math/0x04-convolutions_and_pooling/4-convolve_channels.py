#!/usr/bin/env python3
"""performs a convolution on images with channels:"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """performs a convolution on images with channels:"""
    m, h, w, c = images.shape
    kh, kw, c = kernel.shape
    sh, sw = stride[0], stride[1]

    if padding == 'valid':
        ph = pw = 0
    elif padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + (kh % 2 == 0))
        pw = int((((w - 1) * sw + kw - w) / 2) + (kh % 2 == 0))
    else:
        ph, pw = padding[0], padding[1]

    oh = int(((h + 2 * ph - kh) / sh) + 1)
    ow = int(((w + 2 * pw - kw) / sw) + 1)
    dim = (m, oh, ow)
    out = np.zeros(dim)
    padded_images = np.pad(images,
                           pad_width=((0, 0), (ph, ph), (pw, pw),
                                      (0, 0)), mode='constant',
                           constant_values=0)
    for i in range(dim[1]):
        for j in range(dim[2]):
            x = (i * sh) + kh
            y = (j * sw) + kw
            M = padded_images[:, (i * sh):x, (j * sw):y, :]
            out[:, i, j] = np.tensordot(M, kernel, axes=3)

    return out
