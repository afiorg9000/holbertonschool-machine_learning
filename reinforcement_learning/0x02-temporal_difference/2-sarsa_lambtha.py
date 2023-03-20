#!/usr/bin/env python3
"""performs SARSA(λ):"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """ performs SARSA(λ):"""
    for i in range(episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        for j in range(max_steps):
            new_state, reward, done, info = env.step(action)
            new_action = epsilon_greedy(Q, new_state, epsilon)
            if done:
                Q[state, action] = Q[state, action] + alpha * (
                    reward - Q[state, action])
                break
            else:
                Q[state, action] = Q[state, action] + alpha * (
                    reward + gamma * Q[new_state, new_action] -
                    Q[state, action])
                state = new_state
                action = new_action
        epsilon = epsilon - epsilon_decay if epsilon > min_epsilon else min_epsilon
    return Q
