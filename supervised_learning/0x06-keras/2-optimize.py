#!/usr/bin/env python3
""" Adam optimization for a keras model"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ Adam optimization for a keras model """
    network.compile(optimizer=K.optimizers.Adam(lr=alpha,
                                                beta_1=beta1,
                                                beta_2=beta2),
                    loss='categorical_crossentropy',
                    metrics=['metrics.categorical_accuracy'])
    return None
