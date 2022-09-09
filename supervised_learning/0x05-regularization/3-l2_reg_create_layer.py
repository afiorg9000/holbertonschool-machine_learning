#!/usr/bin/env python3
"""creates a tensorflow layer that includes L2 regularization:"""
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a tensorflow layer that includes L2 regularization:"""
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                        mode=("fan_avg"))
    kernel_regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=initializer,
                            kernel_regularizer=kernel_regularizer,
                            name='layer')
    return layer(prev)