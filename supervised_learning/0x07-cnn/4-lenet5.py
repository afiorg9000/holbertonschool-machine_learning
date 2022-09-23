#!/usr/bin/env python3
"""LeNet-5 architecture using tensorflow:"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """LeNet-5 architecture using tensorflow:"""
    init = tf.keras.initializers.VarianceScaling()
    conv1 = tf.layers.Conv2D(filters=6,
                             kernel_size=5,
                             padding="same",
                             kernel_initializer=init,
                             activation=tf.nn.relu)(x)

    pool1 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                   strides=2)(conv1)
    conv2 = tf.layers.Conv2D(filters=16,
                             kernel_size=5,
                             padding="valid",
                             kernel_initializer=init,
                             activation=tf.nn.relu)(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=[2, 2],
                                   strides=2)(conv2)
    flat1 = tf.layers.Flatten()(pool2)
    fully1 = tf.layers.Dense(units=120,
                             activation=tf.nn.relu,
                             kernel_initializer=init)(flat1)
    fully2 = tf.layers.Dense(units=84,
                             activation=tf.nn.relu,
                             kernel_initializer=init)(fully1)
    out = tf.layers.Dense(units=10,
                          kernel_initializer=init)(fully2)
    loss = tf.losses.softmax_cross_entropy(y, out)
    out_softmax = tf.nn.softmax(out)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    pred = tf.argmax(y, 1)
    val = tf.argmax(out, 1)
    equality = tf.equal(pred, val)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return out_softmax, optimizer, loss, accuracy
