#!/usr/bin/env python3
"""defines a single neuron performing binary classification:"""
import numpy as np


class Neuron:
    """defines a single neuron performing binary classification:"""

    def __init__(self, nx):
        """This code is initializing the weights and bias of a neuron."""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be positive')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter for weight"""
        return self.__W

    @property
    def b(self):
        """getter for bias"""
        return self.__b

    @property
    def A(self):
        """getter for activation"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        C = -(1 / m)*np.sum(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))
        return C

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        self. forward_prop(X)
        cost = self.cost(Y, self.__A)
        prediction = np.where(self.__A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dZ = A - Y
        self.__W -= (alpha * ((1 / m) * np.matmul(X, dZ.T))).T
        self.__b -= (alpha * ((1 / m) * np.sum(dZ)))
