#!/usr/bin/env python3
"""calculate the attention for machine translation"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """SelfAttention"""
    def __init__(self, units):
        """Public instance method"""
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

        super(SelfAttention, self).__init__()

    def call(self, s_prev, hidden_states):
        """Public instance method"""
        s_prev = tf.expand_dims(s_prev, 1)
        score = tf.nn.tanh(self.W(s_prev) + self.U(hidden_states))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights
