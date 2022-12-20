#!/usr/bin/env python3
"""creates a variational autoencoder:"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder:"""
    # encoder
    encoder_inputs = Input(shape=(input_dims,))
    x = encoder_inputs
    for layer in hidden_layers:
        x = Dense(layer, activation='relu')(x)
    z_mean = Dense(latent_dims, activation=None)(x)
    z_log_var = Dense(latent_dims, activation=None)(x)
    z = Lambda(sampling, output_shape=(latent_dims,))([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # decoder
    latent_inputs = Input(shape=(latent_dims,))
    x = latent_inputs
    for layer in hidden_layers[::-1]:
        x = Dense(layer, activation='relu')(x)
    decoder_outputs = Dense(input_dims, activation='sigmoid')(x)
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')
    decoder.summary()

    # autoencoder
    outputs = decoder(encoder(encoder_inputs)[2])
    auto = Model(encoder_inputs, outputs, name='autoencoder')
    auto.summary()

    # loss
    reconstruction_loss = binary_crossentropy(encoder_inputs, outputs)
    reconstruction_loss *= input_dims
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
