#!/usr/bin/env python3
"""calculates the expectation step in the EM algorithm for a GMM:"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """calculates the expectation step in the EM algorithm for a GMM:"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if type(m) is not np.ndarray or m.ndim != 2:
        return None, None
    if type(S) is not np.ndarray or S.ndim != 3:
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    post = np.zeros((k, n))
    for i in range(k):
        post[i] = pi[i] * pdf(X, m[i], S[i])
    marginal = np.sum(post, axis=0)
    likelihood = np.sum(np.log(marginal))
    return post / marginal, likelihood