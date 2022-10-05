#!/usr/bin/env python3
"""ResNet-50 architecture"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ResNet-50 architecture"""
    inititializer = K.initializers.he_normal()
    activ1 = "relu"
    Y = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7),
                            strides=(2, 2), padding='same',
                            kernel_initializer=inititializer)(Y)
    normal1 = K.layers.BatchNormalization()(conv1)
    activ2 = K.layers.Activation(activ1)(normal1)
    max_pool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                      padding='same')(activ2)
    conv2x1 = projection_block(max_pool1, [64, 64, 256], 1)
    conv2x2 = identity_block(conv2x1, [64, 64, 256])
    conv2x3 = identity_block(conv2x2, [64, 64, 256])
    conv3x1 = projection_block(conv2x3, [128, 128, 512])
    conv3x2 = identity_block(conv3x1, [128, 128, 512])
    conv3x3 = identity_block(conv3x2, [128, 128, 512])
    conv3x4 = identity_block(conv3x3, [128, 128, 512])
    conv4x1 = projection_block(conv3x4, [256, 256, 1024])
    conv4x2 = identity_block(conv4x1, [256, 256, 1024])
    conv4x3 = identity_block(conv4x2, [256, 256, 1024])
    conv4x4 = identity_block(conv4x3, [256, 256, 1024])
    conv4x5 = identity_block(conv4x4, [256, 256, 1024])
    conv4x6 = identity_block(conv4x5, [256, 256, 1024])
    conv5x1 = projection_block(conv4x6, [512, 512, 2048])
    conv5x2 = identity_block(conv5x1, [512, 512, 2048])
    conv5x3 = identity_block(conv5x2, [512, 512, 2048])
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                         padding='valid')(conv5x3)
    softmax = K.layers.Dense(units=1000, activation='softmax',
                             kernel_initializer=inititializer)(avg_pool)
    model = K.Model(inputs=Y, outputs=softmax)
    return model
