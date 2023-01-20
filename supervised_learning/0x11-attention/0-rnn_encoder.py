#!/usr/bin/env python3
"""encode for machine translation:"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """encode for machine translation:"""

    def __init__(self, vocab, embedding, units, batch):
        """encode for machine translation:"""
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        super(RNNEncoder, self).__init__()

    def initialize_hidden_state(self):
        """Public instance method"""
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Public instance method"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
