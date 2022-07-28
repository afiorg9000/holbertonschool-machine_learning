#!/usr/bin/env python3
"""adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """return a new matrix"""
    added = []
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            row.append(mat1[i][j] + mat2[i][j])
        added.append(row)
    return added
