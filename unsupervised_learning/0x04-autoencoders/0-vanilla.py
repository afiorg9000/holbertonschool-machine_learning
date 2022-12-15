#!/usr/bin/env python3
"""creates an autoencoder:"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates an autoencoder:"""
    input_encoder = keras.Input(shape=(input_dims,))

    # Encoder
    encoder_layer = input_encoder
    for i in range(len(hidden_layers)):
        encoder_layer = keras.layers.Dense(hidden_layers[i],
                                           activation='relu')(encoder_layer)

    latent = keras.layers.Dense(latent_dims,
                                activation='relu')(encoder_layer)
    encoder = keras.Model(inputs=input_encoder, outputs=latent)

    # Decoder

    input_decoder = keras.Input(shape=(latent_dims,))
    decoder_layer = input_decoder
    for i in range(len(hidden_layers) - 1, -1, -1):
        decoder_layer = keras.layers.Dense(hidden_layers[i],
                                           activation='relu')(decoder_layer)

    output_decoder = keras.layers.Dense(input_dims,
                                        activation='sigmoid')(decoder_layer)

    decoder = keras.Model(inputs=input_decoder, outputs=output_decoder)

    # Autoencoder
    auto = keras.Model(inputs=input_encoder,
                       outputs=decoder(encoder(input_encoder)))
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
