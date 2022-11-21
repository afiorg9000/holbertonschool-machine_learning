#!/usr/bin/env python3
"""determines the probability of a markov chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """determines the probability of a markov chain"""
    if not isinstance(t, int) or t < 1:
        return None
    n = P.shape[0]

    if len(s.shape) != 2 or s.shape[0] != 1 or s.shape[1] != n:
        return None

    p = np.zeros((1, n))
    product = np.linalg.matrix_power(P, t)
    return np.matmul(s, product)
