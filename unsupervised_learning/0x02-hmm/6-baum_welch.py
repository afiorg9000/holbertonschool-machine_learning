#!/usr/bin/env python3
"""performs the Baum-Welch algorithm for a hidden markov model:"""
import numpy as np


def validator(Observations, Transition, Emission, Initial, iterations=1000):
    """performs the Baum-Welch algorithm for a hidden markov model:"""
    if not isinstance(Observations, np.ndarray) or\
            len(Observations.shape) != 1:
        return False
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return False

    T, = Observations.shape
    N, M = Emission.shape

    if not isinstance(Transition, np.ndarray) or Transition.shape != (N, N):
        return False
    if np.any(np.sum(Transition, axis=1) != 1):
        return False
    if not isinstance(Initial, np.ndarray) or Initial.shape != (N, 1):
        return False
    if np.sum(Initial) != 1:
        return False
    if not isinstance(iterations, int) or iterations < 1:
        return False
    return True


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """performs the Baum-Welch algorithm for a hidden markov model:"""
    if validator(Observations, Transition, Emission, Initial, iterations):
        return Transition, Emission
    else:
        return None, None
