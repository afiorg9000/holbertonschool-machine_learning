#!/usr/bin/env python3
"""calculates the Shannon entropy and P affinities relative to a data point:"""
import numpy as np


def HP(Di, beta):
    """calculate HI"""
    Hi = 0
    for i in range(Di.shape[0]):
        Hi += np.exp(-beta * Di[i])
    Hi = - beta * Hi
    """calculate PI"""
    Pi = np.zeros(Di.shape[0])
    for i in range(Di.shape[0]):
        Pi[i] = np.exp(-beta * Di[i]) / Hi
    return (Hi, Pi)
