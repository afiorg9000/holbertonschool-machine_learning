#!/usr/bin/env python3
"""Saves and load model"""
import tensorflow.keras as K


def save_model(network, filename):
    """saves an entire model"""
    K.models.save_model(model=network, filepath=filename)
    return None


def load_model(filename):
    """loads an entire model:"""
    return K.models.load_model(filepath=filename)
