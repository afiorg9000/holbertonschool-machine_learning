#!/usr/bin/env python3
"""concatenates two arrays"""


def cat_arrays(arr1, arr2):
    """return a new list"""
    concatenated = []
    for i in range(len(arr1)):
        concatenated.append(arr1[i])
    for j in range(len(arr2)):
        concatenated.append(arr2[j])
    return concatenated
