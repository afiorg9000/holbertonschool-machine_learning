#!/usr/bin/env python3
"""train an agent that can play Atari’s Breakout:"""
import keras as K
import numpy as np
import rl
import gym
from PIL import Image

build_model = __import__('train_1').build_model
AtariProcessor = __import__('train_1').AtariProcessor
env = gym.make('Breakout-v0')
env.reset()
window_length = 4
nb_actions = env.action_space.n
model = build_model(window_length, nb_actions)
processor = AtariProcessor()
memory = rl.memory.SequentialMemory(limit=1000000, window_length=window_length)


policy = rl.policy.LinearAnnealedPolicy(rl.policy.EpsGreedyQPolicy(), attr='eps',
                                        value_max=1,
                                        value_min=0.1,
                                        value_test=0.05,
                                        nb_steps=1000000)

dqn = rl.agents.dqn.DQNAgent(model=model, nb_actions=nb_actions,
                             policy=policy,
                             memory=memory,
                             processor=processor,
                             nb_steps_warmup=50000,
                             gamma=0.99,
                             train_interval=4,
                             delta_clip=1)

dqn.compile(optimizer=K.optimizers.Adam(lr=0.00025), metrics=['mse'])
dqn.load_weights('policy.h5')
dqn.test(env, nb_episodes=20, visualize=True)
