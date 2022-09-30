#!/usr/bin/env python3
"""dense block"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """dense block"""
    initializer = K.initializers.he_normal()
    activ1 = "relu"

    for i in range(layers):
        normal1 = K.layers.BatchNormalization()(X)
        activ2 = K.layers.Activation(activ1)(normal1)
        filt = 4 * growth_rate
        bottleneck = K.layers.Conv2D(filters=filt, kernel_size=(1, 1),
                                     padding='same',
                                     kernel_initializer=initializer)(activ1)

        normal2 = K.layers.BatchNormalization()(bottleneck)
        activ2 = K.layers.Activation(activ1)(normal2)
        conv = K.layers.Conv2D(filters=growth_rate, kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=initializer)(activ2)

        X = K.layers.concatenate([X, conv])
        nb_filters += growth_rate
    return X, nb_filters