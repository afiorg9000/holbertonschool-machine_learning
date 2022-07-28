#!/usr/bin/env python3
from numpy import concatenate


def cat_arrays(arr1, arr2):
    concatenate = []
    for i in range(len(arr1)):
        concatenate.append(arr1[i])
    for j in range(len(arr2)):
        concatenate.append(arr2[j])
    return concatenate
