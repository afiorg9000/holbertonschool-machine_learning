#!/usr/bin/env python3
"""calculates the sensitivity for each class in a confusion matrix:"""
import numpy as np


def sensitivity(confusion):
    """calculates the sensitivity for each class in a confusion matrix:"""
    return np.diag(confusion) / np.sum(confusion, axis=1)
