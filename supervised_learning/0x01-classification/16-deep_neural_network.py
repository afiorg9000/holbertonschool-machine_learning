#!/usr/bin/env python3
"""deep neural network performing binary classification"""
import numpy as np


class DeepNeuralNetwork:
    """deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """This code is initializing the deep neural network"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be positive integer')
        if type(layers) is not list or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        if  < 1:
            raise ValueError('nodes must be a positive integer')

        self.W1 = np.random.normal(nodes, nx)
        self.b1 = np.zeros(nodes, 1)
        self.A1 = 0
        self.W2 = np.random.normal(1, nodes)
        self.b2 = 0
        self.A2 = 0
