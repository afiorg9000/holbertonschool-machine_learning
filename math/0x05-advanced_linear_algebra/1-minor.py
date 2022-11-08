#!/usr/bin/env python3
"""calculates the minor matrix of a matrix:"""


def squared_minor(m, i, j):
    """calculates the minor of a squared matrix"""
    return [row[:j] + row[j+1:] for row in (m[:i] + m[i+1:])]


def determinant(matrix):
    """calculates the determinant of a matrix:"""

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if not all([isinstance(row, list) for row in matrix]):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]
    square = [len(row) == len(matrix) for row in matrix]

    if not all(square):
        raise ValueError('matrix must be a square matrix')

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    deter = 0
    for c in range(len(matrix)):
        deter += (
            (-1) ** c) * matrix[0][c] * determinant(
                squared_minor(matrix, 0, c))
    return deter


def minor(matrix):
    """calculates the minor matrix of a matrix:"""

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if not all([isinstance(row, list) for row in matrix]):
        raise TypeError("matrix must be a list of lists")

    if not all([len(row) == len(matrix) for row in matrix]):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return [[1]]

    if len(matrix) == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

    result = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix[i])):
            minor_row.append(determinant(squared_minor(matrix, i, j)))
        result.append(minor_row)
    return result
