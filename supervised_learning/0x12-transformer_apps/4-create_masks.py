#!/usr/bin/env python3
"""tensorflow wrapper for the encode instance method"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """tensorflow wrapper for the encode instance method"""
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    decoder_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    size = target.get_shape().as_list()[1]
    combined_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    combined_mask = combined_mask[:, tf.newaxis, :, tf.newaxis]
    combined_mask = combined_mask & tf.cast(tf.linalg.band_part(tf.ones((size, size)), -1, 0), tf.float32)
    combined_mask = combined_mask[:, :, tf.newaxis, :]

    return encoder_mask, combined_mask, decoder_mask
