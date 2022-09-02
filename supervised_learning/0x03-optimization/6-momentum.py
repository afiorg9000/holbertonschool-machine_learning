#!/usr/bin/env python3
"""training operation for a neural network"""
import tensorflow.compat.v1 as tf


def create_momentum_op(loss, alpha, beta1):
    """training operation for a neural network"""
    opt = tf.train.MomentumOptimizer(alpha, beta1)
    return opt.minimize(loss)