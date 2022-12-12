#!/usr/bin/env python3
"""calculates the symmetric P affinities of a data set:"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """Calculates the symmetric P affinities of a data set"""
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)
    for i in range(n):
        low, high = None, None
        Di = np.delete(D[i], i, axis=0)
        Hi, Pi = HP(Di, betas[i])
        while (np.abs(Hi - perplexity) > tol):
            if Hi > perplexity:
                low = betas[i]
                if high is None:
                    betas[i] *= 2
                else:
                    betas[i] = (betas[i] + high) / 2
            else:
                high = betas[i]
                if low is None:
                    betas[i] /= 2
                else:
                    betas[i] = (betas[i] + low) / 2
            Hi, Pi = HP(Di, betas[i])
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = Pi
    P = (P + P.T) / (2 * n)
    return P
