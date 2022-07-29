#!/usr/bin/env python3
"""concatenates two matrices along a specific axis"""
import numpy as np


def cat_matrices2D(mat1, mat2, axis=0):
    """return a new matrix"""
    if axis == 0:
        return np.concatenate((mat1, mat2), axis=axis).tolist()
    elif axis == 1:
        return np.concatenate((mat1, mat2), axis=axis).tolist()
    else:
        return None
