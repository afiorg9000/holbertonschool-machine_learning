#!/usr/bin/env python3
"""performs K-means on a dataset:"""
import sklearn.cluster


def kmeans(X, k):
    """K-means algorithm"""
    k_mean = sklearn.cluster.KMeans(n_clusters=k)
    k_mean.fit(X)
    clss = k_mean.labels_
    C = k_mean.cluster_centers_
    return C, clss
