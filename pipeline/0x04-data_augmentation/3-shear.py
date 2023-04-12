#!/usr/bin/env python3
"""randomly shears an image:"""
import tensorflow as tf

def shear_image(image, intensity):
    """randomly shears an image:"""
    # Generate a random shear angle
    shear_angle = tf.random.uniform([], -intensity, intensity)

    # Define the transform matrix
    transform_matrix = tf.stack(
        [
            [1.0, tf.math.tan(shear_angle * tf.constant(3.141592653589793 / 180.0)), 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Apply the transform matrix to the image
    sheared_image = tf.raw_ops.AffineImage(
        images=image[None],  # Add batch dimension
        transform=transform_matrix,
        output_shape=tf.shape(image),
        interpolation="BILINEAR",
    )

    return sheared_image[0]
