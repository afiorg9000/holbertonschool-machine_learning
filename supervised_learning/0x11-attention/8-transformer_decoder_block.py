#!/usr/bin/env python3
"""create an encoder block for a transformer:"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """create an encoder block for a transformer:"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """create an encoder block for a transformer:"""
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """create an encoder block for a transformer:"""
        attention, attention_block = self.mha1(x, x, x, look_ahead_mask)
        attention = self.dropout1(attention, training=training)
        out1 = self.layernorm1(attention + x)
        attention2, attn_weights_block2 = self.mha2(out1, encoder_output,
                                                    encoder_output,
                                                    padding_mask)
        attention2 = self.dropout2(attention2, training=training)
        out2 = self.layernorm2(attention2 + out1)
        hidden_output = self.dense_hidden(out2)
        output_output = self.dense_output(hidden_output)
        ffn_output = self.dropout3(output_output, training=training)
        output = self.layernorm3(ffn_output + out2)
        return output
