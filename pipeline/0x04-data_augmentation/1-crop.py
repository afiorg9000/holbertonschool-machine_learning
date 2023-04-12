#!/usr/bin/env python3
"""performs a random crop of an image:"""
import tensorflow as tf

def crop_image(image, size):
    """performs a random crop of an image:"""
    # Get the height and width of the image
    height, width, _ = image.shape

    # Get the crop size
    crop_height, crop_width, _ = size

    # Calculate the maximum crop offsets
    max_offset_height = tf.subtract(height, crop_height)
    max_offset_width = tf.subtract(width, crop_width)

    # Generate a random crop offset
    offset_height = tf.random.uniform([], 0, max_offset_height, dtype=tf.int32)
    offset_width = tf.random.uniform([], 0, max_offset_width, dtype=tf.int32)

    # Crop the image
    cropped_image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, crop_height, crop_width)

    return cropped_image
