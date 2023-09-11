# 0x06 Multivariate Probability

> Multivariate probability is the probability distribution on all possible outputs considering any given number of random variables

At the end of this project I was able to answer these conceptual questions:

* Who is Carl Friedrich Gauss?
* What is a joint/multivariate distribution?
* What is a covariance?
* What is a correlation coefficient?
* What is a covariance matrix?
* What is a multivariate Gaussian distribution?

## Tasks

0. Write a function `def mean_cov(X):` that calculates the mean and covariance of a data set:

    * `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set:
        * `n` is the number of data points
        * `d` is the number of dimensions in each data point
        * If `X` is not a 2D `numpy.ndarray`, raise a `TypeError` with the message `X must be a 2D numpy.ndarray`
        * If `n` is less than 2, raise a `ValueError` with the message `X must contain multiple data points`
    * Returns: `mean`, `cov`:
        * `mean` is a `numpy.ndarray` of shape `(1, d)` containing the mean of the data set
        * `cov` is a `numpy.ndarray` of shape `(d, d)` containing the covariance matrix of the data set
    * You are not allowed to use the function `numpy.cov`

1. Write a function `def correlation(C):` that calculates a correlation matrix:

    * `C` is a `numpy.ndarray` of shape `(d, d)` containing a covariance matrix
        * `d` is the number of dimensions
        * If `C` is not a `numpy.ndarray`, raise a `TypeError` with the message `C must be a numpy.ndarray`
        * If `C` does not have shape `(d, d)`, raise a `ValueError` with the message `C must be a 2D square matrix`
    * Returns a `numpy.ndarray` of shape `(d, d)` containing the correlation matrix

2. Create the class `MultiNormal` that represents a Multivariate Normal distribution:

    * class constructor `def __init__(self, data):`
        * `data` is a `numpy.ndarray` of shape `(d, n)` containing the data set:
        * `n` is the number of data points
        * `d` is the number of dimensions in each data point
        * If `data` is not a 2D `numpy.ndarray`, raise a `TypeError` with the message `data must be a 2D numpy.ndarray`
        * If `n` is less than 2, raise a `ValueError` with the message `data must contain multiple data points`
    * Set the public instance variables:
        * `mean` - a `numpy.ndarray` of shape `(d, 1)` containing the mean of `data`
        * `cov` - a `numpy.ndarray` of shape `(d, d)` containing the covariance matrix `data`
    * You are not allowed to use the function `numpy.cov`

3. Update the class `MultiNormal`:

    * public instance method `def pdf(self, x):` that calculates the PDF at a data point:
        * `x` is a `numpy.ndarray` of shape `(d, 1)` containing the data point whose PDF should be calculated
            * `d` is the number of dimensions of the `MultiNormal` instance
        * If `x` is not a `numpy.ndarray`, raise a `TypeError` with the message `x must be a numpy.ndarray`
        * If `x` is not of shape `(d, 1)`, raise a `ValueError` with the message `x must have the shape ({d}, 1)`
        * Returns the value of the PDF
        * You are not allowed to use the function `numpy.cov`
