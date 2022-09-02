#!/usr/bin/env python3
"""gradient descent with momentum optimization algorithm:"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """gradient descent with momentum optimization algorithm:"""
    momentum = beta1 * v + (1 - beta1) * grad
    update = var - alpha * momentum
    return momentum, update
