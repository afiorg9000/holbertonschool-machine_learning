#!/usr/bin/env python3
"""updates a variable in place using the Adam optimization"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ a variable in place using the Adam optimization"""
    # Exponentially Weighted Averages (momentum)
    v = beta1 * v + (1 - beta1) * grad
    # RMSProp
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    # bias correction
    v_corrected = v / (1 - (beta1 ** t))
    s_corrected = s / (1 - (beta2 ** t))
    # variable update with ADAM (adaptive moment estimation)
    var = var - alpha * v_corrected / (s_corrected ** (1/2) + epsilon)
    return var, v, s
