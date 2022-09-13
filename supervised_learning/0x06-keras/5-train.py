#!/usr/bin/env python3
"""trains a model using mini-batch gradient descent:"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """ model using mini-batch gradient descent:"""
    iteration = network.fit(data, labels, batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose, shuffle=shuffle,
                            validation_data=validation_data)
    return iteration
