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
        self.__W = np.random.normal(0.0, 1 / np.sqrt(nx))
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
        Z = np.dot(self.__W * X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        return -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))
