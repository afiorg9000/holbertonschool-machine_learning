# 0x03 Hyperparameter Tuning

> In machine learning, hyperparameter optimization or tuning is the problem of choosing a set of optimal hyperparameters for a learning algorithm. A hyperparameter is a parameter whose value is used to control the learning process

At the end of this project I was able to answer these conceptual questions:

* What is Hyperparameter Tuning?
* What is random search? grid search?
* What is a Gaussian Process?
* What is a mean function?
* What is a Kernel function?
* What is Gaussian Process Regression/Kriging?
* What is Bayesian Optimization?
* What is an Acquisition function?
* What is Expected Improvement?
* What is Knowledge Gradient?
* What is Entropy Search/Predictive Entropy Search?
* What is GPy?
* What is GPyOpt?

## Tasks

0. Create the class `GaussianProcess` that represents a noiseless 1D Gaussian process:

    * Class constructor: `def __init__(self, X_init, Y_init, l=1, sigma_f=1)`:
        * `X_init` is a `numpy.ndarray` of shape `(t, 1)` representing the inputs already sampled with the black-box function
        * `Y_init` is a `numpy.ndarray` of shape `(t, 1)` representing the outputs of the black-box function for each input in `X_init`
        * `t` is the number of initial samples
        * `l` is the length parameter for the kernel
        * `sigma_f` is the standard deviation given to the output of the black-box function
        * Sets the public instance attributes `X`, `Y`, `l`, and `sigma_f` corresponding to the respective constructor inputs
        * Sets the public instance attribute `K`, representing the current covariance kernel matrix for the Gaussian process
    * Public instance method `def kernel(self, X1, X2):` that calculates the covariance kernel matrix between two matrices:
        * `X1` is a `numpy.ndarray` of shape `(m, 1)`
        * `X2` is a `numpy.ndarray` of shape `(n, 1)`
        * the kernel should use the Radial Basis Function (RBF)
        * Returns: the covariance kernel matrix as a `numpy.ndarray` of shape `(m, n)`

1. Based on `0-gp.py`, update the class `GaussianProcess`:

    * Public instance method `def predict(self, X_s):` that predicts the mean and standard deviation of points in a Gaussian process:
        * `X_s` is a `numpy.ndarray` of shape `(s, 1)` containing all of the points whose mean and standard deviation should be calculated
            * `s` is the number of sample points
        * Returns: `mu, sigma`
            * `mu` is a `numpy.ndarray` of shape `(s,)` containing the mean for each point in `X_s`, respectively
            * `sigma` is a `numpy.ndarray` of shape `(s,)` containing the variance for each point in `X_s`, respectively

2. Based on `1-gp.py`, update the class `GaussianProcess`:

    * Public instance method `def update(self, X_new, Y_new):` that updates a Gaussian Process:
        * `X_new` is a `numpy.ndarray` of shape `(1,)` that represents the new sample point
        * `Y_new` is a `numpy.ndarray` of shape `(1,)` that represents the new sample function value
        * Updates the public instance attributes `X`, `Y`, and `K`

3. Create the class `BayesianOptimization` that performs Bayesian optimization on a noiseless 1D Gaussian process:

    * Class constructor `def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):`
        * `f` is the black-box function to be optimized
        * `X_init` is a `numpy.ndarray` of shape `(t, 1)` representing the inputs already sampled with the black-box function
        * `Y_init` is a `numpy.ndarray` of shape `(t, 1)` representing the outputs of the black-box function for each input in `X_init`
        * `t` is the number of initial samples
        * `bounds` is a tuple of `(min, max)` representing the bounds of the space in which to look for the optimal point
        * `ac_samples` is the number of samples that should be analyzed during acquisition
        * `l` is the length parameter for the kernel
        * `sigma_f` is the standard deviation given to the output of the black-box function
        * `xsi` is the exploration-exploitation factor for acquisition
        * `minimize` is a `bool` determining whether optimization should be performed for minimization (`True`) or maximization (`False`)
        * Sets the following public instance attributes:
            * `f`: the black-box function
            * `gp`: an instance of the class `GaussianProcess`
            * `X_s`: a `numpy.ndarray` of shape `(ac_samples, 1)` containing all acquisition sample points, evenly spaced between `min` and `max`
            * `xsi`: the exploration-exploitation factor
            * `minimize`: a `bool` for minimization versus maximization
    * You may use `GP = __import__('2-gp').GaussianProcess`

4. Based on `3-bayes_opt.py`, update the class `BayesianOptimization`:

    * Public instance method `def acquisition(self):` that calculates the next best sample location:
        * Uses the Expected Improvement acquisition function
        * Returns: `X_next, EI`
            * `X_next` is a `numpy.ndarray` of shape `(1,)` representing the next best sample point
            * `EI` is a `numpy.ndarray` of shape `(ac_samples,)` containing the expected improvement of each potential sample
    * You may use `from scipy.stats import norm`

5. Based on `4-bayes_opt.py`, update the class `BayesianOptimization`:

    * Public instance method `def optimize(self, iterations=100):` that optimizes the black-box function:
        * `iterations` is the maximum number of iterations to perform
        * If the next proposed point is one that has already been sampled, optimization should be stopped early
        * Returns: `X_opt, Y_opt`
            * `X_opt` is a `numpy.ndarray` of shape `(1,)` representing the optimal point
            * `Y_opt` is a `numpy.ndarray` of shape `(1,)` representing the optimal function value

