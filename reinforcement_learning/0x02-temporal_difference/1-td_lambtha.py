#!/usr/bin/env python3
"""performs the TD(λ) algorithm:"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """performs the TD(λ) algorithm:"""
    for episode in range(episodes):
        state = env.reset()
        done = False
        E = np.zeros(V.shape)
        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, info = env.step(action)
            E[state] += 1
            delta = reward + gamma * V[next_state] - V[state]
            V[state] += alpha * delta * E[state]
            E[state] = gamma * lambtha * E[state]
            state = next_state
            if done:
                break
    return V
