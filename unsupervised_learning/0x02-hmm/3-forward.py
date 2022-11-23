#!/usr/bin/env python3
"""performs the forward algorithm for a hidden markov model:"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model:"""
    T = Observation.shape[0]
    N, M = Emission.shape
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, T):
        for s in range(N):
            F[s, t] = np.sum(
                F[:,
                  t - 1] * Transition[:,
                                      s] * Emission[s, Observation[t]])
    P = np.sum(F[:, T - 1])
    return P, F
