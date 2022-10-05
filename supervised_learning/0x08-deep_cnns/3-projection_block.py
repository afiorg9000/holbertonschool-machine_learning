#!/usr/bin/env python3
"""3. Projection Block"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """3. Projection Block"""
    F11, F3, F12 = filters
    initializer = K.initializers.HeNormal()
    conv1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=s,
                            padding='same',
                            kernel_initializer=initializer)(A_prev)
    batch1 = K.layers.BatchNormalization()(conv1)
    act1 = K.layers.Activation('relu')(batch1)
    conv2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same',
                            kernel_initializer=initializer)(act1)
    batch2 = K.layers.BatchNormalization()(conv2)
    act2 = K.layers.Activation('relu')(batch2)
    conv3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same',
                            kernel_initializer=initializer)(act2)
    batch2 = K.layers.BatchNormalization()(conv3)
    conv4 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=s,
                            padding='same',
                            kernel_initializer=initializer)(A_prev)
    batch3 = K.layers.BatchNormalization()(conv4)
    add = K.layers.Add()([batch2, batch3])
    act2 = K.layers.Activation('relu')(add)

    return act2
