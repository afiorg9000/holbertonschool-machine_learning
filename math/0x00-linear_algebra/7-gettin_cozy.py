#!/usr/bin/env python3
"""concatenates two matrices along a specific axis"""
import numpy as np


def cat_matrices2D(mat1, mat2, axis=0):
    """return a new matrix"""
    if axis == 0:
        new = [row.copy() for row in mat1] + mat2.copy()
        return new
    elif axis == 1:
        new = [mat1[tmp] + mat2[tmp] for tmp in range(len(mat1))]
        return new
    else:
        return None
