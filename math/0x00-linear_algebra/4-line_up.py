#!/usr/bin/env python3
"""adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """return a new list"""
    added = []
    if len(arr1) != len(arr2):
        return None
    for tmp in range(len(arr1)):
        added.append(arr1[tmp] + arr2[tmp])
    return added
