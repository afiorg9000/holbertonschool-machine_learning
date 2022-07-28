#!/usr/bin/env python3
"""performs matrix multiplication"""
import numpy as np


def mat_mul(mat1, mat2):
    """performs matrix multiplication"""
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    if mat1.shape[1] != mat2.shape[0]:
        return None
    else:
        return np.matmul(mat1, mat2).tolist()
