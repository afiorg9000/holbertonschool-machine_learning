#!/usr/bin/env python3
"""calculates the expectation step in the EM algorithm for a GMM:"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM:"""
    n, d = X.shape
    k = pi.shape[0]
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    g = np.zeros((k, n))
    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])
    marginal = np.sum(g, axis=0)
    l = np.sum(np.log(marginal))
    return g / marginal, l
