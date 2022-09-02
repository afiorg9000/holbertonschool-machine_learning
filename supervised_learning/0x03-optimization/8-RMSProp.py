#!/usr/bin/env python3
"""training operation using the RMSProp optimization algorithm:"""
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """training operation for a neural network in tensorflow"""
    opt = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                    decay=beta2, epsilon=epsilon)
    step_opt = opt.minimize(loss)
    return step_opt
