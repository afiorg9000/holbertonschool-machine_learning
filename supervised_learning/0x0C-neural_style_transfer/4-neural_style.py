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
        self.generate_features()

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

    @staticmethod
    def gram_matrix(input_layer):
        """
        :param input_layer: an instance of tf.Tensor or
            tf.Variable of shape (1, h, w, c)containing the
            layer output whose gram matrix should be calculated
        :return:
        """
        e = 'input_layer must be a tensor of rank 4'
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) \
                or len(input_layer.shape) != 4:
            raise TypeError(e)

        # We make the image channels first
        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        gram = tf.expand_dims(gram, axis=0)
        return gram / tf.cast(n, tf.float32)

    def generate_features(self):
        """ extracts the features used to calculate neural style cost"""

        vgg19 = tf.keras.applications.vgg19

        content_image_input = vgg19.preprocess_input(self.content_image * 255)
        style_image_input = vgg19.preprocess_input(self.style_image * 255)

        content_img_output = self.model(content_image_input)
        style_img_output = self.model(style_image_input)

        list_gram = []
        for out in style_img_output[:-1]:
            list_gram = list_gram + [self.gram_matrix(out)]

        self.gram_style_features = list_gram
        self.content_feature = content_img_output[-1]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculates the style cost for a single layer
        :param style_output: tf.Tensor of shape (1, h, w, c)
            containing the layer style output of the generated image
        :param gram_target: tf.Tensor of shape (1, c, c)
            the gram matrix of the target style output for that layer
        :return:
        """
        err = 'style_output must be a tensor of rank 4'
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError(err)

        c = int(style_output.shape[-1])
        err = 'gram_target must be a tensor of shape [1, {}, {}]'.format(c, c)
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != (1, c, c)):
            raise TypeError(err)

        gram_style = self.gram_matrix(style_output)

        layer_style_cost = tf.reduce_mean(tf.square(gram_style - gram_target))

        return layer_style_cost