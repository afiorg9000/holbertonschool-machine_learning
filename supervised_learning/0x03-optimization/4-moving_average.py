#!/usr/bin/env python3
"""calculates the weighted moving average of a data set:"""
import numpy as np


def moving_average(data, beta):
    """calculates the weighted moving average of a data set:"""
    avg = []
    n = 0
    for i in range(len(data)):
        n = beta * n + (1 - beta) * data[i]
        avg.append(n / (1 - beta ** (i + 1)))
    return avg
