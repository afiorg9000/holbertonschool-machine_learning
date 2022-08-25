#!/usr/bin/env python3
"""calculates the accuracy of a prediction:"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
create_layer = __import__('1-create_layer').create_layer


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction:"""
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

    return accuracy
