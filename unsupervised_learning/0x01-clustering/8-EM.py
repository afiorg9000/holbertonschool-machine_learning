#!/usr/bin/env python3
"""performs the expectation maximization for a GMM:"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """performs the expectation maximization for a GMM:"""
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None, None, None, None
    if type(k) is not int or int(k) != k or k < 1:
        return None, None, None, None, None
    if type(
        iterations) is not int or int(
            iterations) != iterations or iterations < 1:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    l_old = 0
    for i in range(iterations):
        g, lo = expectation(X, pi, m, S)
        if verbose is True and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".
                  format(i, lo.round(5)))
        if np.abs(lo - l_old) <= tol:
            break
        pi, m, S = maximization(X, g)
    g, lL = expectation(X, pi, m, S)
    l_old = lL
    if verbose is True:
        print("Log Likelihood after {} iterations: {}".format(i, lo.round(5)))
    return pi, m, S, g, lL
