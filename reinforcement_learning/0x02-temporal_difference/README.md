# 0x02 Temporal Difference

> TD learning is an unsupervised technique to predict a variable's expected value in a sequence of states. TD uses a mathematical trick to replace complex reasoning about the future with a simple learning procedure that can produce the same results. Instead of calculating the total future reward, TD tries to predict the combination of immediate reward and its own reward prediction at the next moment in time

At the end of this project I was able to answer these conceptual questions:

* What is Monte Carlo?
* What is Temporal Difference?
* What is bootstrapping?
* What is n-step temporal difference?
* What is TD(λ)?
* What is an eligibility trace?
* What is SARSA? SARSA(λ)? SARSAMAX?
* What is ‘on-policy’ vs ‘off-policy’?

## Tasks

0. Write the function `def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):` that performs the Monte Carlo algorithm:

    * `env` is the openAI environment instance
    * `V` is a `numpy.ndarray` of shape `(s,)` containing the value estimate
    * `policy` is a function that takes in a state and returns the next action to take
    * `episodes` is the total number of episodes to train over
    * `max_steps` is the maximum number of steps per episode
    * `alpha` is the learning rate
    * `gamma` is the discount rate
    * Returns: `V`, the updated value estimate

1. Write the function `def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):` that performs the TD(λ) algorithm:

    * `env` is the openAI environment instance
    * `V` is a `numpy.ndarray` of shape `(s,)` containing the value estimate
    * `policy` is a function that takes in a state and returns the next action to take
    * `lambtha` is the eligibility trace factor
    * `episodes` is the total number of episodes to train over
    * `max_steps` is the maximum number of steps per episode
    * `alpha` is the learning rate
    * `gamma` is the discount rate
    * Returns: `V`, the updated value estimate

2. Write the function `def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):` that performs SARSA(λ):

    * `env` is the openAI environment instance
    * `Q` is a `numpy.ndarray` of shape `(s,a)` containing the Q table
    * `lambtha` is the eligibility trace factor
    * `episodes` is the total number of episodes to train over
    * `max_steps` is the maximum number of steps per episode
    * `alpha` is the learning rate
    * `gamma` is the discount rate
    * `epsilon` is the initial threshold for epsilon greedy
    * `min_epsilon` is the minimum value that epsilon should decay to
    * `epsilon_decay` is the decay rate for updating epsilon between episodes
    * Returns: `Q`, the updated Q table

