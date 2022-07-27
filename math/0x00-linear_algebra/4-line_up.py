#!/usr/bin/env python3

def add_arrays(arr1, arr2):
    added = []
    if len(arr1) != len(arr2):
        return None
    for tmp in range(len(arr1)):
        added.append(arr1[tmp] + arr2[tmp])
    return added
