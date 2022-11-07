#!/usr/bin/env python3
"""represents a Multivariate Normal distribution:"""
import numpy as np


class MultiNormal:
    def __init__(self, data):
        if type(data) != np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        elif data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')

        sample_count = data.shape[1]
        self.mean = np.mean(data, axis=1, keepdims=True)
        mu = self.mean
        self.cov = np.matmul(data - mu, data.T - mu.T) / (sample_count - 1)

