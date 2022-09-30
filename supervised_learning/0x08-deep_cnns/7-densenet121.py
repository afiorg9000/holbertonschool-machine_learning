#!/usr/bin/env python3
"""DenseNet-121"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """DenseNet-121"""
    initializer = K.initializers.he_normal()
    activ1 = "relu"
    Y = K.Input(shape=(224, 224, 3))
    filt = 2 * growth_rate
    normal1 = K.layers.BatchNormalization()(Y)
    activ2 = K.layers.Activation(activ1)(normal1)
    conv1 = K.layers.Conv2D(filters=filt, kernel_size=(7, 7), strides=(2, 2),
                             padding='same', kernel_initializer=initializer)(activ2)
    max_pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')(conv1)
    dense1, filt = dense_block(max_pool1, filt, growth_rate, 6)
    trans1, filt = transition_layer(dense1, filt, compression)
    dense2, filt = dense_block(trans1, filt, growth_rate, 12)
    trans2, filt = transition_layer(dense2, filt, compression)
    dense3, filt = dense_block(trans2, filt, growth_rate, 24)
    trans3, filt = transition_layer(dense3, filt, compression)
    dense4, filt = dense_block(trans3, filt, growth_rate, 16)
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                         padding='valid')(dense4)
    softmax = K.layers.Dense(units=1000, activation='softmax',
                             kernel_initializer=initializer)(avg_pool)
    model = K.Model(inputs=Y, outputs=softmax)
    return model
