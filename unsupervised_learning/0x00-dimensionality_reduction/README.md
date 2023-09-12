# 0x00 Dimensionality Reduction

> Dimensionality reduction is the transformation of data from a high-dimensional space into a low-dimensional space so that the low-dimensional representation retains some meaningful properties of the original data.

At the end of this project I was able to answer these conceptual questions:

* What is eigendecomposition?
* What is singular value decomposition?
* What is the difference between eig and svd?
* What is dimensionality reduction and what are its purposes?
* What is principal components analysis (PCA)?
* What is t-distributed stochastic neighbor embedding (t-SNE)?
* What is a manifold?
* What is the difference between linear and non-linear dimensionality reduction?
* Which techniques are linear/non-linear?

## Tasks

0. Write a function `def pca(X, var=0.95):` that performs PCA on a dataset:

    * `X` is a `numpy.ndarray` of shape `(n, d)` where:
        * `n` is the number of data points
        * `d` is the number of dimensions in each point
        * all dimensions have a mean of 0 across all data points
    * `var` is the fraction of the variance that the PCA transformation should maintain
    * Returns: the weights matrix, `W`, that maintains `var` fraction of `X`‘s original variance
    * `W` is a `numpy.ndarray` of shape `(d, nd)` where `nd` is the new dimensionality of the transformed `X`

1. Write a function `def pca(X, ndim):` that performs PCA on a dataset:

    * `X` is a `numpy.ndarray` of shape `(n, d)` where:
        * `n` is the number of data points
        * `d` is the number of dimensions in each point
    * `ndim` is the new dimensionality of the transformed `X`
    * Returns: `T`, a `numpy.ndarray` of shape `(n, ndim)` containing the transformed version of `X`

