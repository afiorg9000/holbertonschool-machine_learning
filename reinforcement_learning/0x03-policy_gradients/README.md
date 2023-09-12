# 0x03 Policy Gradients

> Type of reinforcement learning techniques that rely upon optimizing parametrized policies with respect to the expected return (long-term cumulative reward) by gradient descent. They do not suffer from many of the problems that have been marring traditional reinforcement learning approaches such as the lack of guarantees of a value function, the intractability problem resulting from uncertain state information and the complexity arising from continuous states & actions.

At the end of this project I was able to answer these conceptual questions:

* What is Policy?
* How to calculate a Policy Gradient?
* What and how to use a Monte-Carlo policy gradient?

## Tasks

0. Write a function that computes to policy with a weight of a matrix

    * Prototype: `def policy(matrix, weight):`

1. By using the previous function created `policy`, write a function that computes the Monte-Carlo policy gradient based on a state and a weight matrix

    * Prototype: `def policy_gradient(state, weight):`
        * `state`: matrix representing the current observation of the environment
        * `weight`: matrix of random weight
    * Return: the action and the gradient (in this order)

2. By using the previous function created `policy_gradient`, write a function that implements a full training

    * Prototype: `def train(env, nb_episodes, alpha=0.000045, gamma=0.98):`
        * `env`: initial environment
        * `nb_episodes`: number of episodes used for training
        * `alpha`: the learning rate
        * `gamma`: the discount factor
    * Return: all values of the score (sum of all rewards during one episode loop)

3. Update the prototype of the `train` function by adding a last optional parameter `show_result` (default: `False`). When this parameter is `True`, render the environment every 1000 episodes computed

