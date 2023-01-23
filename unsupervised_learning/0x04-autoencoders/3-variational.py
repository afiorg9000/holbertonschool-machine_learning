#!/usr/bin/env python3
"""creates a variational autoencoder:"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """creates a variational autoencoder:"""
        def sampling(args):
        """Sampling"""
        z_mean, z_log_sigma = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.int_shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(z_log_sigma / 2) * epsilon

    def kl_reconstruction_loss(true, pred):
        """Loss reconstruction"""
        reconstruction_loss = keras.losses.binary_crossentropy(model_input,
                                                               outputs)
        reconstruction_loss *= input_dims
        exp = keras.backend.exp(z_log_sigma)
        kl_loss = 1 + z_log_sigma - keras.backend.square(z_mean) - exp
        kl_loss = keras.backend.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return keras.backend.mean(reconstruction_loss + kl_loss)

    """Encoder"""
    model_input = keras.layers.Input(shape=(input_dims,))
    encoded = model_input
    for layer in hidden_layers:
        encoded = keras.layers.Dense(layer, activation='relu')(encoded)
    z_mean = keras.layers.Dense(latent_dims)(encoded)
    z_log_sigma = keras.layers.Dense(latent_dims)(encoded)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_sigma])
    encoder = keras.Model(model_input, [z, z_mean, z_log_sigma])
    """Decoder"""
    decoded = keras.layers.Input(shape=(latent_dims,))
    input_d = decoded
    for layer in reversed(hidden_layers):
        decoded = keras.layers.Dense(layer,  activation='relu')(decoded)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    decoder = keras.models.Model(input_d, decoded)
    """auto"""
    outputs = decoder(encoder(model_input))
    auto = keras.models.Model(model_input, outputs)
    auto.compile(optimizer='adam', loss=kl_reconstruction_loss)
    return (encoder, decoder, auto)
