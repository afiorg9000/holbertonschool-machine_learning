#!/usr/bin/env python3
"""transition layer"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """transition layer"""
    inititializer = K.initializers.he_normal()
    activ1 = "relu"
    filt = int(nb_filters * compression)

    normal1 = K.layers.BatchNormalization()(X)
    activ2 = K.layers.Activation(activ1)(normal1)
    conv1 = K.layers.Conv2D(filters=filt, kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=inititializer)(activ1)
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                         padding='same')(conv1)
    return avg_pool, filt
