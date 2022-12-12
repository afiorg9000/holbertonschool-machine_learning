#!/usr/bin/env python3
"""calculates the Shannon entropy and P affinities relative to a data point:"""
import numpy as np


def HP(Di, beta):
    """calculate HI"""
    Pi = np.exp(- Di * beta)
    Pi = Pi / np.sum(Pi)
    Hi = np.sum(-Pi * np.log2(Pi))
    return Hi, Pi
