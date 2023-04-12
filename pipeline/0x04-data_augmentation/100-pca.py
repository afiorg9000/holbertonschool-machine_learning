#!/usr/bin/env python3
"""performs PCA color augmentation as described"""
import tensorflow as tf
import numpy as np

def pca_color(image, alphas):
    """performs PCA color augmentation as described"""
    # Flatten the image tensor into a 2D tensor with shape (height*width, channels)
    orig_shape = tf.shape(image)
    image = tf.reshape(image, [-1, 3])

    # Compute the mean of the image tensor
    mean = tf.reduce_mean(image, axis=0)

    # Subtract the mean from the image tensor
    centered = image - mean

    # Compute the covariance matrix of the centered image tensor
    cov = tfp.stats.covariance(centered)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigvals, eigvecs = tf.linalg.eigh(cov)

    # Generate a random vector of length 3 with normal distribution
    rnd = tf.random.normal([3], stddev=alphas)

    # Compute the color shift vector using the eigenvectors, eigenvalues, and random vector
    color_shift = tf.reduce_sum(tf.multiply(eigvecs, tf.reshape(rnd * eigvals, [1, 3])), axis=1)

    # Apply the color shift to the image tensor
    augmented = tf.add(image, color_shift)
    augmented = tf.reshape(augmented, orig_shape)

    # Clip the augmented image tensor to the valid range of pixel values
    augmented = tf.clip_by_value(augmented, 0, 255)

    return augmented
