#!/usr/bin/env python3
"""performs tasks for neural style transfer:"""
import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class NST:
    """performs tasks for neural style transfer:"""
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """performs tasks for neural style transfer:"""

        if type(style_image) is not np.ndarray:
            raise TypeError('style_image must be a numpy.ndarray with shape (h, w, 3)')
        if style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError('style_image must be a numpy.ndarray with shape (h, w, 3)')

        if type(content_image) is not np.ndarray:
            raise TypeError('content_image must be a numpy.ndarray with shape (h, w, 3)')
        if content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError('content_image must be a numpy.ndarray with shape (h, w, 3)')

        if type(alpha) is not int and type(alpha) is not float or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        if type(beta) is not int and type(beta) is not float or beta < 0:
            raise TypeError('beta must be a non-negative number')

        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()

    @staticmethod
    def scale_image(image):
        """rescales an image such that its pixels values
        are between 0 and 1 and its largest side is 512 pixels"""
        if type(image) is not np.ndarray:
            raise TypeError('image must be a numpy.ndarray with shape (h, w, 3)')
        if image.ndim != 3 or image.shape[2] != 3:
            raise TypeError('image must be a numpy.ndarray with shape (h, w, 3)')
        h, w, c = image.shape
        if h > w:
            new_h = 512
            new_w = int((new_h * w) / h)
        else:
            new_w = 512
            new_h = int((new_w * h) / w)
        image = np.expand_dims(image, axis=0)
        image = tf.image.resize_bicubic(image, (new_h, new_w))
        image = tf.cast(image, tf.float32)
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)
        return image

    def load_model(self):
        """creates the model used to calculate cost"""
        vgg_model = tf.keras.applications.vgg19.VGG19(
            include_top=False,
            weights='imagenet')
        vgg_model.trainable = False

        style_outputs = [vgg_model.get_layer(name).output
                         for name in self.style_layers]
        content_outputs = [vgg_model.get_layer(self.content_layer).output]

        model_outputs = style_outputs + content_outputs

        return (tf.keras.models.Model(vgg_model.input, model_outputs))
    