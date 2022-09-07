#!/usr/bin/env python3
"""updates the weights and biases of a neural network using gradient descent with L2 regularization:"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """updates the weights and biases of a neural network using gradient descent with L2 regularization:"""
    for i in range(0, L):
  first_derivative = -2 * np.dot(X.T, y - np.dot(X, weights)) + 2 * alpha * weights
  weights = weights - (alpha / L) * first_derivative
return weigh