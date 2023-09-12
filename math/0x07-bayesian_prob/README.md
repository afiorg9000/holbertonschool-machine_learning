# 0x07 Bayesian Probability

> Bayesian probability is an interpretation of the concept of probability, in which, instead of frequency or propensity of some phenomenon, probability is interpreted as reasonable expectation representing a state of knowledge or as quantification of a personal belief.

At the end of this project I was able to answer these conceptual questions:

* What is Bayesian Probability?
* What is Bayes’ rule and how do you use it?
* What is a base rate?
* What is a prior?
* What is a posterior?
* What is a likelihood?

## Tasks

You are conducting a study on a revolutionary cancer drug and are looking to find the probability that a patient who takes this drug will develop severe side effects. During your trials, `n` patients take the drug and `x` patients develop severe side effects. You can assume that `x` follows a binomial distribution.

0. Write a function `def likelihood(x, n, P):` that calculates the likelihood of obtaining this data given various hypothetical probabilities of developing severe side effects:

    * `x` is the number of patients that develop severe side effects
    * `n` is the total number of patients observed
    * `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
    * If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
    * If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
    * If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
    * If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
    * If any value in `P` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in P must be in the range [0, 1]`
    * Returns: a 1D `numpy.ndarray` containing the likelihood of obtaining the data, `x` and `n`, for each probability in `P`, respectively

1. Based on `0-likelihood.py`, write a function `def intersection(x, n, P, Pr):` that calculates the intersection of obtaining this data with the various hypothetical probabilities:

    * `x` is the number of patients that develop severe side effects
    * `n` is the total number of patients observed
    * `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
    * `Pr` is a 1D `numpy.ndarray` containing the prior beliefs of `P`
    * If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
    * If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
    * If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
    * If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
    * If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message `Pr must be a numpy.ndarray with the same shape as P`
    * If any value in `P` or `Pr` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in {P} must be in the range [0, 1]` where `{P}` is the incorrect variable
    * If `Pr` does not sum to `1`, raise a `ValueError` with the message `Pr must sum to 1` **Hint: use [numpy.isclose](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.isclose.html)**
    * All exceptions should be raised in the above order
    * Returns: a 1D `numpy.ndarray` containing the intersection of obtaining `x` and `n` with each probability in `P`, respectively

2. Based on `1-intersection.py`, write a function `def marginal(x, n, P, Pr):` that calculates the marginal probability of obtaining the data:

    * `x` is the number of patients that develop severe side effects
    * `n` is the total number of patients observed
    * `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of patients developing severe side effects
    * `Pr` is a 1D `numpy.ndarray` containing the prior beliefs about `P`
    * If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
    * If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
    * If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
    * If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
    * If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message `Pr must be a numpy.ndarray with the same shape as P`
    * If any value in `P` or `Pr` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in {P} must be in the range [0, 1]` where `{P}` is the incorrect variable
    * If `Pr` does not sum to `1`, raise a `ValueError` with the message `Pr must sum to 1`
    * All exceptions should be raised in the above order
    * Returns: the marginal probability of obtaining `x` and `n`

3. Based on `2-marginal.py`, write a function `def posterior(x, n, P, Pr):` that calculates the posterior probability for the various hypothetical probabilities of developing severe side effects given the data:

    * `x` is the number of patients that develop severe side effects
    * `n` is the total number of patients observed
    * `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
    * `Pr` is a 1D `numpy.ndarray` containing the prior beliefs of `P`
    * If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
    * If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
    * If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
    * If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
    * If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message `Pr must be a numpy.ndarray with the same shape as P`
    * If any value in `P` or `Pr` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in {P} must be in the range [0, 1]` where `{P}` is the incorrect variable
    * If `Pr` does not sum to `1`, raise a `ValueError` with the message `Pr must sum to 1`
    * All exceptions should be raised in the above order
    * Returns: the posterior probability of each probability in `P` given `x` and `n`, respectively

4. Based on `3-posterior.py`, write a function `def posterior(x, n, p1, p2):` that calculates the posterior probability that the probability of developing severe side effects falls within a specific range given the data:

    * `x` is the number of patients that develop severe side effects
    * `n` is the total number of patients observed
    * `p1` is the lower bound on the range
    * `p2` is the upper bound on the range
    * You can assume the prior beliefs of `p` follow a uniform distribution
    * If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
    * If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
    * If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
    * If `p1` or `p2` are not floats within the range `[0, 1]`, raise a `ValueError` with the message `{p} must be a float in the range [0, 1]` where `{p}` is the corresponding variable
    * if `p2` <= `p1`, raise a `ValueError` with the message `p2 must be greater than p1`
    * The only import you are allowed to use is `from scipy import special`
    * Returns: the posterior probability that `p` is within the range `[p1, p2]` given `x` and `n`


