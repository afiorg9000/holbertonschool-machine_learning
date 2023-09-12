# 0x01 Clustering

> Cluster analysis or clustering is the task of grouping a set of objects in such a way that objects in the same group (called a cluster) are more similar (in some sense) to each other than to those in other groups (clusters). It is a main task of exploratory data analysis, and a common technique for statistical data analysis, used in many fields, including pattern recognition, image analysis, information retrieval, bioinformatics, data compression, computer graphics and machine learning.

At the end of this project I was able to answer these conceptual questions:

* What is a multimodal distribution?
* What is a cluster?
* What is cluster analysis?
* What is “soft” vs “hard” clustering?
* What is K-means clustering?
* What are mixture models?
* What is a Gaussian Mixture Model (GMM)?
* What is the Expectation-Maximization (EM) algorithm?
* How to implement the EM algorithm for GMMs
* What is cluster variance?
* What is the mountain/elbow method?
* What is the Bayesian Information Criterion?
* How to determine the correct number of clusters
* What is Hierarchical clustering?
* What is Agglomerative clustering?
* What is Ward’s method?
* What is Cophenetic distance?
* What is scikit-learn?
* What is scipy?

## Tasks

0. Write a function `def initialize(X, k):` that initializes cluster centroids for K-means:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset that will be used for K-means clustering
        * `n` is the number of data points
        * `d` is the number of dimensions for each data point
    * `k` is a positive integer containing the number of clusters
    * The cluster centroids should be initialized with a **multivariate uniform distribution** along each dimension in `d`:
        * The minimum values for the distribution should be the minimum values of `X` along each dimension in `d`
        * The maximum values for the distribution should be the maximum values of `X` along each dimension in `d`
        * You should use `numpy.random.uniform` exactly once
    * You are not allowed to use any loops
    * Returns: a `numpy.ndarray` of shape `(k, d)` containing the initialized centroids for each cluster, or `None` on failure

1. Write a function `def kmeans(X, k, iterations=1000):` that performs K-means on a dataset:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset
        * `n` is the number of data points
        * `d` is the number of dimensions for each data point
    * `k` is a positive integer containing the number of clusters
    * `iterations` is a positive integer containing the maximum number of iterations that should be performed
    * If no change in the cluster centroids occurs between iterations, your function should return
    * Initialize the cluster centroids using a multivariate uniform distribution (based on `0-initialize.py`)
    * If a cluster contains no data points during the update step, reinitialize its centroid
    * You should use `numpy.random.uniform` exactly twice
    * You may use at most 2 loops
    * Returns: `C, clss`, or `None, None` on failure
        * `C` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for each cluster
        * `clss` is a `numpy.ndarray` of shape `(n,)` containing the index of the cluster in `C` that each data point belongs to

2. Write a function `def variance(X, C):` that calculates the total intra-cluster variance for a data set:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
    * `C` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for each cluster
    * You are not allowed to use any loops
    * Returns: `var`, or `None` on failure
        * `var` is the total variance

3. Write a function `def optimum_k(X, kmin=1, kmax=None, iterations=1000):` that tests for the optimum number of clusters by variance:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
    * `kmin` is a positive integer containing the minimum number of clusters to check for (inclusive)
    * `kmax` is a positive integer containing the maximum number of clusters to check for (inclusive)
    * `iterations` is a positive integer containing the maximum number of iterations for K-means
    * This function should analyze *at least* 2 different cluster sizes
    * You should use:
        * `kmeans = __import__('1-kmeans').kmeans`
        * `variance = __import__('2-variance').variance`
    * You may use at most 2 loops
    * Returns: `results, d_vars`, or `None, None` on failure
        * `results` is a list containing the outputs of K-means for each cluster size
        * `d_vars` is a list containing the difference in variance from the smallest cluster size for each cluster size

4. Write a function `def initialize(X, k):` that initializes variables for a Gaussian Mixture Model:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
    * `k` is a positive integer containing the number of clusters
    * You are not allowed to use any loops
    * Returns: `pi, m, S`, or `None, None, None` on failure
        * `pi` is a `numpy.ndarray` of shape `(k,)` containing the priors for each cluster, initialized evenly
        * `m` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for each cluster, initialized with K-means
        * `S` is a `numpy.ndarray` of shape `(k, d, d)` containing the covariance matrices for each cluster, initialized as identity matrices
    * You should use `kmeans = __import__('1-kmeans').kmeans`

5. Write a function `def pdf(X, m, S):` that calculates the probability density function of a Gaussian distribution:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the data points whose PDF should be evaluated
    * `m` is a `numpy.ndarray` of shape `(d,)` containing the mean of the distribution
    * `S` is a `numpy.ndarray` of shape `(d, d)` containing the covariance of the distribution
    * You are not allowed to use any loops
    * You are not allowed to use the function `numpy.diag` or the method `numpy.ndarray.diagonal`
    * Returns: `P`, or `None` on failure
        * `P` is a `numpy.ndarray` of shape `(n,)` containing the PDF values for each data point
    * All values in `P` should have a minimum value of `1e-300`

6. Write a function `def expectation(X, pi, m, S):` that calculates the expectation step in the EM algorithm for a GMM:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
    * `pi` is a `numpy.ndarray` of shape `(k,)` containing the priors for each cluster
    * `m` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for each cluster
    * `S` is a `numpy.ndarray` of shape `(k, d, d)` containing the covariance matrices for each cluster
    * You may use at most 1 loop
    * Returns: `g, l`, or `None, None` on failure
        * `g` is a `numpy.ndarray` of shape `(k, n)` containing the posterior probabilities for each data point in each cluster
        * `l` is the total log likelihood
    * You should use `pdf = __import__('5-pdf').pdf`

7. Write a function `def maximization(X, g):` that calculates the maximization step in the EM algorithm for a GMM:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
    * `g` is a `numpy.ndarray` of shape `(k, n)` containing the posterior probabilities for each data point in each cluster
    * You may use at most 1 loop
    * Returns: `pi, m, S`, or `None, None, None` on failure
        * `pi` is a `numpy.ndarray` of shape `(k,)` containing the updated priors for each cluster
        * `m` is a `numpy.ndarray` of shape `(k, d)` containing the updated centroid means for each cluster
        * `S` is a `numpy.ndarray` of shape `(k, d, d)` containing the updated covariance matrices for each cluster

8. Write a function `def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):` that performs the expectation maximization for a GMM:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
    * `k` is a positive integer containing the number of clusters
    * `iterations` is a positive integer containing the maximum number of iterations for the algorithm
    * `tol` is a non-negative float containing tolerance of the log likelihood, used to determine early stopping i.e. if the difference is less than or equal to `tol` you should stop the algorithm
    * `verbose` is a boolean that determines if you should print information about the algorithm
        * If `True`, print `Log Likelihood after {i} iterations: {l}` every 10 iterations and after the last iteration
        * `{i}` is the number of iterations of the EM algorithm
        * `{l}` is the log likelihood, rounded to 5 decimal places
    * You should use:
        * `initialize = __import__('4-initialize').initialize`
        * `expectation = __import__('6-expectation').expectation`
        * `maximization = __import__('7-maximization').maximization`
    * You may use at most 1 loop
    * Returns: `pi, m, S, g, l`, or `None, None, None, None, None` on failure
        * `pi` is a `numpy.ndarray` of shape `(k,)` containing the priors for each cluster
        * `m` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for each cluster
        * `S` is a `numpy.ndarray` of shape `(k, d, d)` containing the covariance matrices for each cluster
        * `g` is a `numpy.ndarray` of shape `(k, n)` containing the probabilities for each data point in each cluster
        * `l` is the log likelihood of the model

9. Write a function `def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):` that finds the best number of clusters for a GMM using the Bayesian Information Criterion:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set
    * `kmin` is a positive integer containing the minimum number of clusters to check for (inclusive)
    * `kmax` is a positive integer containing the maximum number of clusters to check for (inclusive)
        * If `kmax` is `None`, `kmax` should be set to the maximum number of clusters possible
    * `iterations` is a positive integer containing the maximum number of iterations for the EM algorithm
    * `tol` is a non-negative float containing the tolerance for the EM algorithm
    * `verbose` is a boolean that determines if the EM algorithm should print information to the standard output
    * You should use `expectation_maximization = __import__('8-EM').expectation_maximization`
    * You may use at most 1 loop
    * Returns: `best_k, best_result, l, b`, or `None, None, None, None` on failure
        * `best_k` is the best value for k based on its BIC
        * `best_result` is tuple containing `pi, m, S`
            * `pi` is a `numpy.ndarray` of shape `(k,)` containing the cluster priors for the best number of clusters
            * `m` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for the best number of clusters
            * `S` is a `numpy.ndarray` of shape `(k, d, d)` containing the covariance matrices for the best number of clusters
        * `l` is a `numpy.ndarray` of shape `(kmax - kmin + 1)` containing the log likelihood for each cluster size tested
        * `b` is a `numpy.ndarray` of shape `(kmax - kmin + 1)` containing the BIC value for each cluster size tested
            * Use: `BIC = p * ln(n) - 2 * l`
            * `p` is the number of parameters required for the model: [number-of-parameters-to-be-learned-in-k-guassian-mixture-model](https://stats.stackexchange.com/questions/436181/number-of-parameters-to-be-learned-in-k-guassian-mixture-model)
            * `n` is the number of data points used to create the model
            * `l` is the log likelihood of the model

10. Write a function `def kmeans(X, k):` that performs K-means on a dataset:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset
    * `k` is the number of clusters
    * The only import you are allowed to use is `import sklearn.cluster`
    * Returns: `C, clss`
        * `C` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means for each cluster
        * `clss` is a `numpy.ndarray` of shape `(n,)` containing the index of the cluster in `C` that each data point belongs to

11. Write a function `def gmm(X, k):` that calculates a GMM from a dataset:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset
    * `k` is the number of clusters
    * The only import you are allowed to use is `import sklearn.mixture`
    * Returns: `pi, m, S, clss, bic`
        * `pi` is a `numpy.ndarray` of shape `(k,)` containing the cluster priors
        * `m` is a `numpy.ndarray` of shape `(k, d)` containing the centroid means
        * `S` is a `numpy.ndarray` of shape `(k, d, d)` containing the covariance matrices
        * `clss` is a `numpy.ndarray` of shape `(n,)` containing the cluster indices for each data point
        * `bic` is a `numpy.ndarray` of shape `(kmax - kmin + 1)` containing the BIC value for each cluster size tested

12. Write a function `def agglomerative(X, dist):` that performs agglomerative clustering on a dataset:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the dataset
    * `dist` is the maximum cophenetic distance for all clusters
    * Performs agglomerative clustering with Ward linkage
    * Displays the dendrogram with each cluster displayed in a different color
    * The only imports you are allowed to use are:
        * `import scipy.cluster.hierarchy`
        * `import matplotlib.pyplot as plt`
    * Returns: `clss`, a `numpy.ndarray` of shape `(n,)` containing the cluster indices for each data point

