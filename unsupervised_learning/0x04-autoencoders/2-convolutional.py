#!/usr/bin/env python3
"""creates a convolutional autoencoder:"""
import tensorflow.keras as K
initializer = K.initializers.he_normal(seed=None)


def autoencoder(input_dims, filters, latent_dims):
    """creates a convolutional autoencoder:"""
    input_encoder = K.Input(shape=input_dims)
    input_decoder = K.Input(shape=latent_dims)

    # Encoder
    encoder = K.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                              padding='same', activation='relu',
                              kernel_initializer=initializer)(input_encoder)
    encoder = K.layers.MaxPooling2D(pool_size=(2, 2),
                                    padding='same')(encoder)
    for i in range(1, len(filters)):
        encoder = K.layers.Conv2D(filters=filters[i], kernel_size=(3, 3),
                                  padding='same', activation='relu',
                                  kernel_initializer=initializer)(encoder)
        encoder = K.layers.MaxPooling2D(pool_size=(2, 2),
                                        padding='same')(encoder)

    # Decoder
    decoder = K.layers.Conv2D(filters=filters[-1], kernel_size=(3, 3),
                              padding='same', activation='relu',
                              kernel_initializer=initializer)(input_decoder)
    decoder = K.layers.UpSampling2D(size=(2, 2))(decoder)
    for i in range(len(filters) - 2, 0, -1):
        decoder = K.layers.Conv2D(filters=filters[i], kernel_size=(3, 3),
                                  padding='same', activation='relu',
                                  kernel_initializer=initializer)(decoder)
        decoder = K.layers.UpSampling2D(size=(2, 2))(decoder)
    decoder = K.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                              padding='valid', activation='relu',
                              kernel_initializer=initializer)(decoder)
    decoder = K.layers.UpSampling2D(size=(2, 2))(decoder)
    decoder = K.layers.Conv2D(filters=input_dims[2], kernel_size=(3, 3),
                              padding='same', activation='sigmoid',
                              kernel_initializer=initializer)(decoder)

    # Autoencoder
    encoder = K.models.Model(inputs=input_encoder, outputs=encoder)
    decoder = K.models.Model(inputs=input_decoder, outputs=decoder)
    input_auto = K.Input(shape=input_dims)
    encoder_out = encoder(input_auto)
    decoder_out = decoder(encoder_out)
    auto = K.models.Model(inputs=input_auto, outputs=decoder_out)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
