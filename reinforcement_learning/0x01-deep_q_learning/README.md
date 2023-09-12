# 0x01 Deep Q-learning

> Deep Q-Learning is a type of reinforcement learning algorithm that uses a deep neural network to approximate the Q-function, which is used to determine the optimal action to take in a given state

At the end of this project I was able to answer these conecptual questions:

* What is Deep Q-learning?
* What is the policy network?
* What is replay memory?
* What is the target network?
* Why must we utilize two separate networks during training?
* What is keras-rl? How do you use it?

## Tasks

0. Write a python script `train.py` that utilizes `keras`, `keras-rl`, and `gym` to train an agent that can play Atari’s Breakout:

    * Your script should utilize `keras-rl`‘s `DQNAgent`, `SequentialMemory`, and `EpsGreedyQPolicy`
    * Your script should save the final policy network as `policy.h5`

    Write a python script `play.py` that can display a game played by the agent trained by `train.py`:

    * Your script should load the policy network saved in `policy.h5`
    * Your agent should use the `GreedyQPolicy`

