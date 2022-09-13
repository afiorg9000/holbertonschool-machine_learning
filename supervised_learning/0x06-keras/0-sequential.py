#!/usr/bin/env python3
"""neural network with keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """neural network with keras"""
    model = K.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[0],
                                     activation=activations[0],
                                     input_shape=(nx,),
                                     kernel_initializer=K.
                                     initializers.he_normal(),
                                     kernel_regularizer=K.
                                     regularizers.l2(lambtha)))
        else:
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=K.
                                     regularizers.l2(lambtha)))
    return model
