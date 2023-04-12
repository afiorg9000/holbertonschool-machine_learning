#!/usr/bin/env python3
"""randomly changes the brightness of an image:"""
import tensorflow as tf

def change_brightness(image, max_delta):
    """randomly changes the brightness of an image:"""
    # Generate a random brightness delta
    brightness_delta = tf.random.uniform([], -max_delta, max_delta)

    # Apply the brightness delta to the image
    brightened_image = tf.image.adjust_brightness(image, brightness_delta)

    return brightened_image
