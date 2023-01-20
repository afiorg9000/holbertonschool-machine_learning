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
        batches = tf.shape(Q)[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        def split_heads(x, batch_size):
            x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
            return tf.transpose(x, perm=[0, 2, 1, 3])
        Q = split_heads(Q, batches)
        K = split_heads(K, batches)
        V = split_heads(V, batches)
        outs, weights = sdp_attention(Q, K, V, mask)
        outs = tf.transpose(outs, perm=[0, 2, 1, 3])
        outs = tf.reshape(outs, [batches, -1, self.dm])
        return self.linear(outs), weights
