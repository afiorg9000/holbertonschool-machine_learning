#!/usr/bin/env python3
""" normalizes (standardizes) a matrix:"""
import numpy as np


def normalize(X, m, s):
    """normalizes (standardizes) a matrix:"""
    return (X - m) / s
