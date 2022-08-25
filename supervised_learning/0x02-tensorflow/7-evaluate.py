#!/usr/bin/env python3
"""evaluates the output of a neural network:"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def evaluate(X, Y, save_path):
    """evaluates the output of a neural network:"""
    with tf.Session() as session:
        save = tf.train.import_meta_graph(save_path + ".meta")
        save.restore(session, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        return session.run((y_pred, accuracy, loss), feed_dict={x: X, y: Y})
