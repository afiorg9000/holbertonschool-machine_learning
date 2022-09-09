#!/usr/bin/env python3
"""layer of a neural network using dropout:"""
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """layer of a neural network using dropout:"""
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode=("fan_avg"))
    dropout = tf.layers.Dropout(rate=(1 - keep_prob))
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=initializer,
                            kernel_regularizer=dropout,
                            name='layer')
    return layer(prev)
