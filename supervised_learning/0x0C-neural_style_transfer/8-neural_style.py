#!/usr/bin/env python3
"""performs tasks for neural style transfer:"""
import numpy as np
import tensorflow as tf


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


        tf.executing_eagerly()
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
        image = tf.image.resize(image, (new_h, new_w))
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

    def style_cost(self, style_outputs):
        """  calculate the style cost """
        len_style_layers = len(self.style_layers)

        if (type(style_outputs) is not list or
                len(style_outputs) != len_style_layers):
            raise TypeError("style_outputs must be a list with a "
                            "length of {}".format(len_style_layers))

        weight = 1 / len_style_layers
        s_cost = 0

        gram_targets = self.gram_style_features

        for s_output, target in zip(style_outputs, self.gram_style_features):
            s_cost += self.layer_style_cost(s_output, target) * weight

        return s_cost


    def content_cost(self, content_output):
        '''Calculates the content cost for the generated image
        Args:
            content_output - a tf.Tensor containing the content output for
                the generated image
        Returns: the content cost
        '''
        content_shape = self.content_feature.shape

        if (not (isinstance(content_output, tf.Tensor) or
                 isinstance(content_output, tf.Variable)) or
                content_output.shape != content_shape):
            raise TypeError("content_output must be a tensor of "
                            "shape {}".format(content_shape))

        h, w, c = content_output.shape[1:]
        square = tf.square(content_output - self.content_feature)
        hwc = tf.cast(h * w * c, tf.float32)

        return tf.reduce_sum(square) / hwc

    def total_cost(self, generated_image):
        '''Calculates the total cost for the generated image
        Args:
            generated_image - a tf.Tensor of shape (1, nh, nw, 3) containing
                the generated image
        Returns: (J, J_content, J_style)
                 - J is the total cost
                 - J_content is the content cost
                 - J_style is the style cost
        '''
        content_image_shape = self.content_image.shape

        if (not (isinstance(generated_image, tf.Tensor) or
                 isinstance(generated_image, tf.Variable)) or
                generated_image.shape != content_image_shape):
            raise TypeError("generated_image must be a tensor of "
                            "shape {}".format(content_image_shape))

        generated_image = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255
        )

        outputs = self.model(generated_image)

        style_outputs = outputs[:-1]
        style_cost = self.style_cost(style_outputs)

        content_output = outputs[-1]
        content_cost = self.content_cost(content_output)

        total_c = style_cost * self.beta + content_cost * self.alpha
        return (total_c, content_cost, style_cost)

    def compute_grads(self, generated_image):
        """
        compute the gradients for the generated image
        :param generated_image: tf.Tensor generated image of
        shape (1, nh, nw, 3)
        :return: gradients, J_total, J_content, J_style
            gradients is a tf.Tensor containing
            the gradients for the generated image
            J_total is the total cost for the generated image
            J_content is the content cost for the generated image
            J_style is the style cost for the generated image
        """
        s = self.content_image.shape
        err = "generated_image must be a tensor of shape {}".format(s)
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)):
            raise TypeError(err)
        if generated_image.shape != s:
            raise TypeError(err)

        with tf.GradientTape() as tape:
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradient = tape.gradient(J_total, generated_image)
        return gradient, J_total, J_content, J_style