#!/usr/bin/env python3
"""Inception Block"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """Inception Block"""
    F1, F3R, F3, F5R, F5, FPP = filters
    conv1 = K.layers.Conv2D(F1, kernel_size=(1, 1), padding='same',
                            activation='relu')(A_prev)

    conv2 = K.layers.Conv2D(F3R, kernel_size=(1, 1), padding='same',
                            activation='relu')(A_prev)

    conv3 = K.layers.Conv2D(F3, kernel_size=(3, 3), padding='same',
                            activation='relu')(conv2)

    conv4 = K.layers.Conv2D(F5R, kernel_size=(1, 1), padding='same',
                            activation='relu')(A_prev)

    conv5 = K.layers.Conv2D(F5, kernel_size=(5, 5), padding='same',
                            activation='relu')(conv4)

    maxpool = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                    padding='same')(A_prev)

    maxpool_conv = K.layers.Conv2D(FPP, kernel_size=(1, 1), padding='same',
                                   activation='relu')(maxpool)

    convolution = K.layers.concatenate([conv1, conv3, conv5, maxpool_conv],
                                         axis=3)

    return convolution