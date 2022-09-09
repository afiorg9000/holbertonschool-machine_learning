#!/usr/bin/env python3
""" Dropout regularization using gradient descent:"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ Dropout regularization using gradient descent:"""
    for i in range(L, 0, -1):
        m = Y.shape[0]
        weight = weights.copy()
        A = cache["A" + str(i)]
        DZ = A - Y
        DZ = np.multiply(
            np.dot(weight["W" + str(i + 1)].T, DZ),
            (1 - np.power(A, 2)))
        DZ = (DZ * cache["D" + str(i)]) / keep_prob

        DW = 1 / m * np.dot(DZ, cache["A" + str(i - 1)].T)
        DB = 1 / m * np.sum(DZ, axis=1, keepdims=True)

        weights["W" + str(i)] = weight[
            "W" + str(i)] - (alpha * DB)
        weights["b" + str(i)] = weight[
            "b" + str(i)] - (alpha * DB)
        return weights
