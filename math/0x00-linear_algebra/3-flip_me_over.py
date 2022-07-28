#!/usr/bin/env python3
"""returns the transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """ returns transpose"""
    transpose = []
    for i in range(len(matrix[0])):
        transpose.append([])
        for j in range(len(matrix)):
            transpose[i].append(matrix[j][i])
    return transpose
