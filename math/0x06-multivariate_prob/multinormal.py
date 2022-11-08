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
