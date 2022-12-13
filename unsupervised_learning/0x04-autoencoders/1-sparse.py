#!/usr/bin/env python3
"""creates a sparse autoencoder:"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """creates a sparse autoencoder:"""
    # encoder
    input_encoder = Input(shape=(input_dims,))
    hidden_encoder = input_encoder
    for i in hidden_layers:
        hidden_encoder = Dense(i, activation='relu')(hidden_encoder)
    latent_encoder = Dense(latent_dims,
                           activation='relu',
                           activity_regularizer=regularizers.
                           l1(lambtha))(hidden_encoder)
    encoder = Model(input_encoder, latent_encoder)

    # decoder
    input_decoder = Input(shape=(latent_dims,))
    hidden_decoder = input_decoder
    for i in hidden_layers[::-1]:
        hidden_decoder = Dense(i, activation='relu')(hidden_decoder)
    output_decoder = Dense(input_dims, activation='sigmoid')(hidden_decoder)
    decoder = Model(input_decoder, output_decoder)

    # autoencoder
    input_auto = Input(shape=(input_dims,))
    encoder_out = encoder(input_auto)
    decoder_out = decoder(encoder_out)
    auto = Model(input_auto, decoder_out)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
