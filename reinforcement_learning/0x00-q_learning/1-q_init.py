#!/usr/bin/env python3
"""initializes the Q-table:"""
import numpy as np


def q_init(env):
    """initializes the Q-table:"""
    return np.zeros((env.observation_space.n, env.action_space.n))
