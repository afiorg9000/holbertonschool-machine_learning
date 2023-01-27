#!/usr/bin/env python3
"""encode for machine translation:"""
import tensorflow as tf
import numpy as np


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

    def positional_encoding(max_seq_len, dm):
        """calculates the positional encoding for a transformer:"""
        PE = np.zeros((max_seq_len, dm))
        even = np.array([x for x in range(0, dm, 2)])
        pos = np.arange(max_seq_len)
        PE[:, ::2] = np.sin(pos[:, np.newaxis] / np.power(10000, even / dm))
        PE[:, 1::2] = np.cos(pos[:, np.newaxis] / np.power(10000, even / dm))
        return PE

    def sdp_attention(Q, K, V, mask=None):
        """calculates the scaled dot product attention:"""
        matmul_qk = tf.matmul(Q, K, transpose_b=True)
        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * (-1e9))
        weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(weights, V)
        return output, weights

class MultiHeadAttention(tf.keras.layers.Layer):
    """perform multi head attention:"""

    def __init__(self, dm, h):
        """perform multi head attention:"""
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """perform multi head attention:"""
        batch_size = tf.shape(Q)[0]
        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = sdp_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1,
                                                         self.dm))
        output = self.linear(concat_attention)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        """perform multi head attention:"""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])


class EncoderBlock(tf.keras.layers.Layer):
    """create an encoder block for a transformer:"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """create an encoder block for a transformer:"""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """create an encoder block for a transformer:"""
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return

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

class Encoder(tf.keras.layers.Layer):
    """create the encoder for a transformer:"""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """create the encoder for a transformer:"""
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, self.dm)
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
                       ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """create the encoder for a transformer:"""
        seq_len = x.shape[1]
        embedding = self.embedding(x)
        embedding *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        embedding += self.positional_encoding[:seq_len]
        encoder_out = self.dropout(embedding, training=training)
        for i in range(self.N):
            encoder_out = self.blocks[i](encoder_out, training, mask)
        return encoder_out

class Decoder(tf.keras.layers.Layer):
    """create the decoder for a transformer:"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """create the decoder for a transformer:"""
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
                       ]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """create the decoder for a transformer:"""
        seq_len = x.shape[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]
        x = self.dropout(x, training=training)
        for i in range(self.N):
            x = self.blocks[i](x, encoder_output, training, look_ahead_mask,
                               padding_mask)
        return x

class Transformer(tf.keras.Model):
    """create a transformer network:"""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """create a transformer network:"""
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target,
                               drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """create a transformer network:"""
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)
        final_output = self.linear(dec_output)
        return final_output
