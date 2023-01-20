#!/usr/bin/env python3
"""decode for machine translation:"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """decode for machine translation:"""
    def __init__(self, vocab, embedding, units, batch):
        """decode for machine translation:"""
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

        super(RNNDecoder, self).__init__()

    def call(self, x, s_prev, hidden_states):
        """decode for machine translation:"""
        _, units = s_prev.shape
        attention = SelfAttention(units)
        context_vector, _ = attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x],
                      axis=-1)
        output, s = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)
        return y, s
