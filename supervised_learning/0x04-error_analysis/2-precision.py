#!/usr/bin/env python3
"""calculates the precision for each class in a confusion matrix:"""
import numpy as np


def precision(confusion):
    """calculates the precision for each class in a confusion matrix:"""
    return np.diag(confusion) / np.sum(confusion, axis=0)
