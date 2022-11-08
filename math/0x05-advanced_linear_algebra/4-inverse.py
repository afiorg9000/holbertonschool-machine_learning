#!/usr/bin/env python3
"""calculates the inverse of a matrix:"""


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


def cofactor(matrix):
    """calculates the cofactor matrix of a matrix:"""

    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if [x for x in matrix if type(x) is not list]:
        raise TypeError("matrix must be a list of lists")

    if [x for x in matrix if len(x) != len(matrix)]:
        raise ValueError("matrix must be a non-empty square matrix")

    new_matrix = minor(matrix).copy()
    n = len(new_matrix)

    for i in range(n):
        for j in range(n):
            if (i + j) % 2 != 0:
                new_matrix[i][j] = new_matrix[i][j] * - 1
    return new_matrix


def adjugate(matrix):
    """calculates the adjugate matrix of a matrix:"""

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if not all([isinstance(row, list) for row in matrix]):
        raise TypeError("matrix must be a list of lists")

    if not all([len(row) == len(matrix) for row in matrix]):
        raise ValueError("matrix must be a non-empty square matrix")

    matrix_cof = cofactor(matrix)
    return [[row[i] for row in matrix_cof] for i in range(len(matrix_cof[0]))]


def inverse(matrix):
    """calculates the inverse of a matrix:"""

    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if [x for x in matrix if type(x) is not list]:
        raise TypeError("matrix must be a list of lists")

    if [x for x in matrix if len(x) != len(matrix)]:
        raise ValueError("matrix must be a non-empty square matrix")

    new_matrix = []
    n = len(matrix)
    new_matrix = adjugate(matrix).copy()
    det = determinant(matrix)

    if det == 0:
        return None
    for i in range(n):
        for j in range(n):
            if new_matrix[i][j] != 0:
                new_matrix[i][j] = new_matrix[i][j] / det
            else:
                new_matrix[i][j] = 0.0
    return new_matrix
