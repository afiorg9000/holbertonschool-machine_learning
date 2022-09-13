#!/usr/bin/env python3
"""trains a model using mini-batch gradient descent:"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ model using mini-batch gradient descent:"""
    stopping = []
    if early_stopping and validation_data:
        stopping.append(K.callbacks.EarlyStopping(patience=patience))
    iteration = network.fit(data, labels, batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose, shuffle=shuffle,
                            validation_data=validation_data,
                            callbacks=stopping)
    return iteration
