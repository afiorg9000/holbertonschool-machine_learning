#!/usr/bin/env python3
"""calculates the shape of a matrix"""


def matrix_shape(matrix):
    """calculates the shape of a matrix"""
    shape = []
    while type(matrix) == list:
        shape.append(len(matrix))
        matrix = matrix[0]
        
    return shape
