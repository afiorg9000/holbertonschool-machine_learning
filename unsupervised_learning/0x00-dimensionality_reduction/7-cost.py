#!/usr/bin/env python3
"""calculates the cost of the t-SNE transformation:"""
import numpy as np


def cost(P, Q):
    """Calculates the cost of the t-SNE transformation"""
    Q = np.maximum(Q, 1e-12)
    P = np.maximum(P, 1e-12)
    C = np.sum(P * np.log(P / Q))
    return C
