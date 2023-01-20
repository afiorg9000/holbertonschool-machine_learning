#!/usr/bin/env python3
"""create a transformer network:"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """create a transformer network:"""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """create a transformer network:"""
        super().__init__()
        self.encoder = Encoder(
            N=N,
            dm=dm,
            h=h,
            hidden=hidden,
            input_vocab=input_vocab,
            max_seq_len=max_seq_input,
            drop_rate=drop_rate,
        )
        self.decoder = Decoder(
            N=N,
            dm=dm,
            h=h,
            hidden=hidden,
            target_vocab=target_vocab,
            max_seq_len=max_seq_target,
            drop_rate=drop_rate,
        )
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """create a transformer network:"""
        encoder_out = self.encoder(
            inputs,
            training,
            encoder_mask,
        )
        decoder_out = self.decoder(
            target,
            encoder_out,
            training,
            look_ahead_mask,
            decoder_mask,
        )

        return self.linear(decoder_out)
