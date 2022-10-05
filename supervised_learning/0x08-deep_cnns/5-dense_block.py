#!/usr/bin/env python3
"""dense block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """dense block"""
    initializer = K.initializers.HeNormal()

    for i in range(layers):
        batch_norm = K.layers.BatchNormalization()(X)
        act = K.layers.Activation('relu')(batch_norm)
        convol = K.layers.Conv2D(filters=4*growth_rate, kernel_size=(1, 1),
                                 padding='same',
                                 kernel_initializer=initializer)(act)
        batch1 = K.layers.BatchNormalization()(convol)
        act1 = K.layers.Activation('relu')(batch1)
        convol1 = K.layers.Conv2D(filters=growth_rate, kernel_size=(3, 3),
                                  padding='same',
                                  kernel_initializer=initializer)(act1)
        X = K.layers.concatenate([X, convol1])
        nb_filters += growth_rate
    return X, nb_filters
