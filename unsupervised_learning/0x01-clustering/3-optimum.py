#!/usr/bin/env python3
"""initializes cluster centroids for K-means:"""
import numpy as np

def initialize(X, k):
    """initializes cluster centroids for K-means:"""
    n = X.shape[0]  # Number of samples in dataset

    randoms = np.random.randint(n, size=k)  # Randomly pick k indices out of n samples

    initialCentroids = X[randoms]  # Get corresponding sample points from dataset
