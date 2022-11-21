#!/usr/bin/env python3
"""hidden markov model:"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """hidden markov model:"""
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
        return None, None
    if Emission.shape[0] != Transition.shape[0]:
        return None, None
    if Initial.shape[0] != Transition.shape[0] or Initial.shape[1] != 1:
        return None, None
    if not np.isclose(np.sum(Emission, axis=1), 1).all():
        return None, None
    if not np.isclose(np.sum(Transition, axis=1), 1).all():
        return None, None
    if not np.isclose(np.sum(Initial, axis=0), 1).all():
        return None, None
    T = Observation.shape[0]
    N, M = Emission.shape
    V = np.zeros((N, T))
    V[:, 0] = Initial.T * Emission[:, Observation[0]]
    for t in range(1, T):
        V[:, t] = np.max(V[:, t - 1] * Transition.T *
                         Emission[np.newaxis, :, Observation[t]].T, axis=1)
    path = [np.argmax(V[:, T - 1])]
    for i in range(T - 2, -1, -1):
        path.append(np.argmax(V[:, i] * Transition[:, path[-1]]))
    path = path[::-1]
    P = np.max(V[:, T - 1])
    return path, P
