#!/usr/bin/env python3
"""represents a noiseless 1D Gaussian process:"""
import numpy as np


class GaussianProcess:
    """represents a noiseless 1D Gaussian process:"""
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix between two matrices"""
        m, n = X1.shape[0], X2.shape[0]
        K = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                K[i, j] = self.sigma_f ** 2 * np.exp(
                    -(X1[i] - X2[j]) ** 2 / (2 * self.l ** 2))
        return K
