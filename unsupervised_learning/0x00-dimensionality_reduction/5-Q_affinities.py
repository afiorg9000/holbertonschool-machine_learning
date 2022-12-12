#!/usr/bin/env python3
"""Calculates the Q affinities"""
import numpy as np


def Q_affinities(Y):
    """Calculates the Q affinities"""
    n, ndim = Y.shape
    sum_Y = np.sum(np.square(Y), 1)
    num = -2 * np.dot(Y, Y.T)
    num = 1 / (1 + np.add(np.add(num, sum_Y).T, sum_Y))
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return Q, num
