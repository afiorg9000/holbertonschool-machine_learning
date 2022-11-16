#!/usr/bin/env python3
"""initializes cluster centroids for K-means:"""
import numpy as np

def initialize(X, k):
    """initializes cluster centroids for K-means:"""
    d = X.shape[1]
    min = np.amin(X,axis=0)
    max = np.amax(X,axis=0)
    centroides = np.random.uniform(low = min, high = max, size = (k,d))
    return centroides