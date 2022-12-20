#!/usr/bin/env python3
"""performs K-means on a dataset:"""
import sklearn.cluster


def kmeans(X, k):
    """K-means algorithm"""
    km = sklearn.cluster.KMeans(n_clusters=k)
    km.fit(X)

    return km.cluster_centers_, km.labels_
