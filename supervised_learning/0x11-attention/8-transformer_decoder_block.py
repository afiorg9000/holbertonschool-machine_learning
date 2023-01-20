#!/usr/bin/env python3
"""create an encoder block for a transformer:"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """create an encoder block for a transformer:"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """create an encoder block for a transformer:"""
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)
        super(DecoderBlock, self).__init__()

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """create an encoder block for a transformer:"""
        start, weights1 = self.mha1(x, x, x, look_ahead_mask)
        start = self.dropout1(start, training=training)
        start = self.layernorm1(x + start)
        mid, weights2 = self.mha2(start, encoder_output, encoder_output,
                                  padding_mask)
        mid = self.dropout2(mid, training=training)
        mid = self.layernorm2(start + mid)
        out = self.dense_hidden(mid)
        out = self.dense_output(out)
        out = self.dropout3(out, training=training)
        out = self.layernorm3(mid + out)
        return out
