#!/usr/bin/env python3
"""performs Q-learning:"""
import numpy as np
import gym


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """performs Q-learning:"""
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_rewards.append(0)
        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)
            if done:
                if reward == 0:
                    reward = -1
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state, :]) - Q[state, action])
            state = new_state
            total_rewards[-1] += reward
            if done:
                break
        epsilon = epsilon * (1 - epsilon_decay)
        epsilon = max(epsilon, min_epsilon)
    return Q, total_rewards
