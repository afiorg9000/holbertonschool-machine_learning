#!/usr/bin/env python3
"""concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """return a new matrix"""
    if len(mat1[0]) == len(mat2[0]):
        return