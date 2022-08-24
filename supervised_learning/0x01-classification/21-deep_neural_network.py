#!/usr/bin/env python3
"""deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """This code is initializing the deep neural network"""
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        for layer in range(self.L):
            weight = "W" + str(layer + 1)
            bias = "b" + str(layer + 1)
            if layer == 0:
                self.__weights[weight] = np.random.randn(layers[layer],
                                                         nx) * np.sqrt(2 / nx)
                self.__weights[bias] = np.zeros((layers[layer], 1))
            else:
                self.__weights[weight] = np.random.randn(
                    layers[layer], layers[layer - 1]) * np.sqrt(
                        2 / layers[layer - 1])
                self.__weights[bias] = np.zeros((layers[layer], 1))

    """getter for layer"""
    @property
    def L(self):
        return(self.__L)

    """getter for cache"""
    @property
    def cache(self):
        return(self.__cache)

    """getter for weight"""
    @property
    def weights(self):
        return(self.__weights)

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        m = X.shape[1]
        for lyr in range(self.__L + 1):
            if lyr == 0:
                self.__cache["A0"] = X
            else:
                Z = np.dot(self.__weights["W" + str(lyr)],
                           self.__cache["A" + str(lyr - 1)]
                           ) + self.__weights["b" + str(lyr)]
                self.__cache["A" + str(lyr)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        C = -(1 / m)*np.sum(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))
        return C

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dZ = cache["A" + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            dW = (1 / m) * np.matmul(dZ, cache["A" + str(i - 1)].T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            dA = cache["A" + str(i - 1)] * (1 - cache["A" + str(i - 1)])
            dZ = np.matmul(self.__weights["W" + str(i)].T, dZ) * dA
            self.__weights["W" + str(i)] = self.weights[
                "W" + str(i)] - (alpha * dW)
            self.__weights["b" + str(i)] = self.__weights[
                "b" + str(i)]-(alpha * db)
