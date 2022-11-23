#!/usr/bin/env python3
"""performs the backward algorithm for a hidden markov model:"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """performs the backward algorithm for a hidden markov model:"""
    T = Observation.shape[0]
    N, M = Emission.shape
    B = np.zeros((N, T))
    B[:, T - 1] = np.ones((N))
    for t in range(T - 2, -1, -1):
        for s in range(N):
            B[s, t] = np.sum(
                B[:,
                  t + 1] * Transition[s,
                                      :] * Emission[:,
                                                    Observation[t + 1]])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0])
    return P, B
