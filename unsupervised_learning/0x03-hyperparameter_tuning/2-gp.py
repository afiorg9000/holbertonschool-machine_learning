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

    def predict(self, X_s):
        """predicts deviation of points in a Gaussian process:"""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu_s = K_s.T.dot(K_inv).dot(self.Y)

        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov_s)
        return mu_s.squeeze(), sigma

    def update(self, X_new, Y_new):
        """updates a Gaussian Process"""
        X_new = np.reshape(X_new, (1, 1))
        Y_new = np.reshape(Y_new, (1, 1))
        self.X = np.append(self.X, X_new, axis=0)
        self.Y = np.append(self.Y, Y_new, axis=0)
        self.K = self.kernel(self.X, self.X)
