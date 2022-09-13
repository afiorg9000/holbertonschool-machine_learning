#!/usr/bin/env python3
"""neural network with keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """neural network with keras"""
    K.Model()
    inputs = K.Input(shape=nx,)
    x = K.layers.Dense(layers[0],
                       activation=activations[0],
                       kernel_regularizer=K.regularizers.l2(lambtha),
                       input_shape=(nx,))(inputs)
    y = x

    for i in range(1, len(layers)):
        if i == 1:
            x = K.layers.Dropout(1 - keep_prob)(x)
        else:
            y = K.layers.Dropout(1 - keep_prob)(y)
        y = K.layers.Dense(layers[i],
                           activation=activations[i],
                           kernel_regularizer=K.regularizers.l2(lambtha))(y)
    model = K.Model(inputs=inputs, outputs=y)
    return model
