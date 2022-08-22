#!/usr/bin/env python3
"""neural network with one hidden layer performing binary classification:"""
import numpy as np


class NeuralNetwork:
    """neural network with one hidden layer"""

    def __init__(self, nx, nodes):
        """This code is initializing the neural network"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be positive integer')
        if type(nodes) is not int:
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.W1 = np.random.normal(nodes, nx)
        self.b1 = np.zeros(nodes, 1)
        self.A1 = 0
        self.W2 = np.random.normal(1, nodes)
        self.b2 = 0
        self.A2 = 0

    @property
    def W1(self):
        """getter for hidden layer weight vector"""
        return self.__W1

    @property
    def b1(self):
        """getter for hidden layer bias"""
        return self.__b1

    @property
    def A1(self):
        """getter for hidden layer activation"""
        return self.__A1

    @property
    def W2(self):
        """getter for output neuron weight vector"""
        return self.__W2

    @property
    def b1(self):
        """getter for output neuron bias"""
        return self.__b2

    @property
    def A2(self):
        """getter for output neuron activation"""
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural netwrk"""
        Z1 = np.dot(self.__W1 * X) + self.__b
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2 * X) + self.__b
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        return -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        self. forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neural network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 1:
            raise ValueError("alpha must be positive")
