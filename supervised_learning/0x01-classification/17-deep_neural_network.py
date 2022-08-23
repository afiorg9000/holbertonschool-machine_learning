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
