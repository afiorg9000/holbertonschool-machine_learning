#!/usr/bin/env python3
"""performs the forward algorithm for a hidden markov model:"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """performs the forward algorithm for a hidden markov model:"""
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape != (N, N) or Initial.shape != (N, 1):
        return None, None
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, T):
        for s in range(N):
            F[s, t] = np.sum(F[:, t - 1] * Transition[:, s] *
                             Emission[s, Observation[t]])
    P = np.sum(F[:, T - 1])
    return P, F
