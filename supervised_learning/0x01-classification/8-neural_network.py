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
