#!/usr/bin/env python3
"""creates a batch normalization layer for a neural network"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """creates a batch normalization layer for a neural network"""
    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
        name="layer")(prev)
    mean, varience = tf.nn.moments(layer, axes=[0])
    gamma = tf.ones([n])
    beta = tf.zeros([n])
    return activation(tf.nn.batch_normalization(
        layer, mean, varience, beta, gamma, 1e-8))
