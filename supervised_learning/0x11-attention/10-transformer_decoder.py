#!/usr/bin/env python3
"""create the decoder for a transformer:"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """create the decoder for a transformer:"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """create the decoder for a transformer:"""
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = self.positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h,
                       hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

        super(Decoder, self).__init__()

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """create the decoder for a transformer:"""
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)

        for i in range(self.N):
            x, block1, block2 = self.blocks[i](x, encoder_output, training,
                                               look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights
