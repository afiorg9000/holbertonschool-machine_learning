#!/usr/bin/env python3
"""changes the hue of an image:"""
import tensorflow as tf

def change_hue(image, delta):
    """changes the hue of an image:"""
    # Apply the hue delta to the image
    hue_image = tf.image.adjust_hue(image, delta)

    return hue_image
