#!/usr/bin/env python3
"""calculates the cost of a neural network with L2 regularization:"""
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """calculates the cost of a neural network with L2 regularization:"""
    return cost + tf.losses.get_regularization_losses()
