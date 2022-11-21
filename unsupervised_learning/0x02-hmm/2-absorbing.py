#!/usr/bin/env python3
"""determines if a markov chain is absorbing:"""
import numpy as np


def absorbing(P):
    """determines if a markov chain is absorbing:"""
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    if np.sum(P, axis=1).all() != 1:
        return False
    if np.diag(P).all() == 1:
        return True
    for i in range(P.shape[0]):
        if np.diag(P)[i] == 1:
            for j in range(P.shape[0]):
                if P[i][j] != 0 and i != j:
                    return False
    diag = np.diag(P)
    ab = (diag == 1)
    if ab.all():
        return True
    for i in range(len(diag)):
        for j in range(len(diag)):
            if P[i, j] > 0 and ab[j]:
                ab[i] = 1
    ab2 = (ab == 1)
    if ab2.all():
        return True
    return False