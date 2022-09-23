#!/usr/bin/env python3
"""modified version of the LeNet-5 architecture using keras:"""
import tensorflow.keras as K


def lenet5(X):
    """modified version of the LeNet-5 architecture using keras:"""
    initializer = K.initializers.HeNormal()
    conv1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                            activation='relu',
                            kernel_initializer=initializer)(X)
    sub1 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                            activation='relu',
                            kernel_initializer=initializer)(sub1)
    sub2 = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    flat = K.layers.Flatten()(sub2)
    dense1 = K.layers.Dense(units=120, activation='relu',
                            kernel_initializer=initializer)(flat)
    dense2 = K.layers.Dense(units=84, activation='relu',
                            kernel_initializer=initializer)(dense1)
    out = K.layers.Dense(units=10, activation='softmax',
                         kernel_initializer=initializer)(dense2)
    model = K.Model(inputs=X, outputs=out)
    adam = K.optimizers.Adam()
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
