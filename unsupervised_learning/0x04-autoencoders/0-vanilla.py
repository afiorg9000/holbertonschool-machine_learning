#!/usr/bin/env python3
import tensorflow as tf
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    # encoder
    encoder_inputs = keras.Input(shape=(input_dims,))
    x = keras.layers.Dense(hidden_layers[0], activation='relu')(encoder_inputs)

    for layer in hidden_layers[1:]:
        x = keras.layers.Dense(layer, activation='relu')(x)
    encoder_outputs = keras.layers.Dense(latent_dims)(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs, name="encoder")

    decoder_inputs = keras.Input(shape=(latent_dims,))
    x = keras.layers.Dense(hidden_layers[-1],
                           activation='relu')(decoder_inputs)

    for layer in reversed(hidden_layers[:-1]):
        x = keras.layers.Dense(layer, activation='relu')(x)
    decoder_outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

    autoencoder_inputs = keras.Input(shape=(input_dims,))
    x = encoder(autoencoder_inputs)
    autoencoder_outputs = decoder(x)
    auto = keras.Model(autoencoder_inputs,
                       autoencoder_outputs, name="autoencoder")

    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
