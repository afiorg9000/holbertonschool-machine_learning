#!/usr/bin/env python3
"""uses epsilon-greedy to determine the next action:"""
import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon):
    """uses epsilon-greedy to determine the next action:"""
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state, :])
