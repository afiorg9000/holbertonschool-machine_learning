#!/usr/bin/env python3
"""creates the forward propagation graph for the neural network:"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the neural network:"""
    h = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        h = create_layer(h, layer_sizes[i], activations[i])
    return h
