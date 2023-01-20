#!/usr/bin/env python3
"""perform multi head attention:"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """perform multi head attention:"""

    def __init__(self, dm, h):
        """perform multi head attention:"""
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

        super(MultiHeadAttention, self).__init__()

    def call(self, Q, K, V, mask):
        """perform multi head attention:"""
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = tf.concat(tf.split(Q, self.h, axis=-1), axis=0)
        K = tf.concat(tf.split(K, self.h, axis=-1), axis=0)
        V = tf.concat(tf.split(V, self.h, axis=-1), axis=0)

        output, weights = sdp_attention(Q, K, V, mask)

        output = tf.concat(tf.split(output, self.h, axis=0), axis=-1)
        output = self.linear(output)

        return output, weights
