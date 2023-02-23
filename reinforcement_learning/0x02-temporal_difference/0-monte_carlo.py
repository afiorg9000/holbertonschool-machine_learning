#!/usr/bin/env python3
"""performs the Monte Carlo algorithm:"""


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """performs the Monte Carlo algorithm:"""
    state_action = set()
    for _ in range(episodes):
        state = env.reset()
        action = policy(state)
        action_reward = [(state, action, None)]
        for _ in range(max_steps):
            state, reward, done, _ = env.step(action)
            action = policy(state)
            action_reward.append((state, action, reward))
            if done:
                break
        T = len(action_reward) - 1
        G = 0
        for a in range(T - 1, -1, -1):
            state, action, _ = action_reward[a]
            _, _, reward_t_1 = action_reward[a + 1]
            G = gamma * G + reward_t_1
            if not (state, action) in state_action:
                V[state] = V[state] + alpha * (G - V[state])
            state_action.add((state, action))
    return V