#!/usr/bin/env python3
"""has the trained agent play an episode:"""
import gym
import numpy as np


def play(env, Q, max_steps=100):
    """has the trained agent play an episode:"""
    state = env.reset()
    env.render()
    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        env.render()
        state = new_state
        if done:
            break
    return reward
