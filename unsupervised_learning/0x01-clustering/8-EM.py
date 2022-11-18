#!/usr/bin/env python3
"""performs the expectation maximization for a GMM:"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """performs the expectation maximization for a GMM:"""
    pi, m, S = initialize(X, k)
    l_old = 0
    for i in range(iterations):
        g, l = expectation(X, pi, m, S)
        if verbose is True and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(i,
                  l.round(5)))
        if np.abs(l - l_old) <= tol:
            break
        pi, m, S = maximization(X, g)
        l_old = l
    g, l = expectation(X, pi, m, S)
    if verbose is True:
        print("Log Likelihood after {} iterations: {}".format(i,
              l.round(5)))
    return pi, m, S, g, l
