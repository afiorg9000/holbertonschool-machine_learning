#!/usr/bin/env python3
"""Returns: the tensor output of the layer"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_layer(prev, n, activation):
    """Returns: the tensor output of the layer"""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, activation=activation,
                                  kernel_initializer=initializer, name='layer')
    return layer(prev)
