#!/usr/bin/env python3
"""creates a learning rate decay operation"""
import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """creates a learning rate decay operation"""
    alpha = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                        decay_rate, staircase=True)
    return alpha
