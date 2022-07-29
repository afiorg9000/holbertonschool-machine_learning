#!/usr/bin/env python3
"""performs matrix multiplication"""
import numpy as np


def mat_mul(mat1, mat2):
    """performs matrix multiplication"""
    if (len(mat1[0]) == len(mat2)):
        return [[sum(a*b for a, b in zip(r, c))
                for c in zip(*mat2)] for r in mat1]
    else:
        return None
