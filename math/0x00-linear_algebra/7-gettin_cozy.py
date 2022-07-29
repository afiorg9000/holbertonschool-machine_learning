#!/usr/bin/env python3
"""concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """return a new matrix"""
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return mat1 + mat2
    if axis == 1 and len(mat1) == len(mat2):
        concatenated = []
        for r1, r2 in zip(mat1, mat2):
            concatenated.append(r1 + r2)
        return concatenated
    return None
