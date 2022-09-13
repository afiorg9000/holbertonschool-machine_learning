#!/usr/bin/env python3
"""Save and Load Configuration"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """saves a model’s configuration in JSON format:"""
    network.save_weights(filepath=filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """loads a model with a specific configuration:"""
    network.load_weights(filename)
    return None
