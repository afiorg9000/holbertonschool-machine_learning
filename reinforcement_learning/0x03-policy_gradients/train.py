#!/usr/bin/env python3
"""function that implements a full training."""
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """function that implements a full training."""
    w = np.random.rand(4, 2)
    totalReward = []

    for episode in range(nb_episodes):
        state = env.reset()[None, :]
        grads = []
        rewards = []
        score = 0

        while True:
            if show_result and (episode % 1000 == 0):
                env.render()

            action, grad = policy_gradient(state, w)
            forwardState, reward, done, _ = env.step(action)
            forwardState = forwardState[None, :]
            grads.append(grad)
            rewards.append(reward)
            score += reward
            state = forwardState
            if done:
                break

        for i in range(len(grads)):
            w += alpha * grads[i] *\
                sum([r * gamma ** r for t, r in enumerate(rewards[i:])])

        totalReward.append(score)
        print("{}: {}".format(episode, score), end="\r", flush=False)

    return totalReward