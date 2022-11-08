#!/usr/bin/env python3
"""represents a Multivariate Normal distribution:"""
import numpy as np


class MultiNormal:
    def __init__(self, data):
        """Constructor Class"""
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
        X_mean = data - self.mean
        self.cov = np.matmul(X_mean, X_mean.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """calculates the PDF at a data point"""
        if type(x) is not np.ndarray or len(x.shape) != 2:
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape[1] != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        if x.shape[0] != self.mean.shape[0]:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        X_mean = x - self.mean
        inv = np.linalg.inv(self.cov)
        det = np.linalg.det(self.cov)
        result = 1 / np.sqrt(((2 * np.pi) ** self.mean.shape[0]) * det)

        exp = np.dot(X_mean.T, np.dot(inv, X_mean))
        pdf = float(result * np.exp(-0.5 * exp))
        return 1
