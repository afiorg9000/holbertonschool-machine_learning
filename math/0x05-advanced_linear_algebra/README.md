# 0x05 Advanced linear algebra

> This project covers the second part of Linear Algebra implementation with Python for ML

At the end of this project, I was able to answer these conceptual questions:

* What is a determinant? How would you calculate it?
* What is a minor, cofactor, adjugate? How would calculate them?
* What is an inverse? How would you calculate it?
* What are eigenvalues and eigenvectors? How would you calculate them?
* What is definiteness of a matrix? How would you determine a matrix’s definiteness?

## Tasks

0. Write a function `def determinant(matrix):` that calculates the determinant of a matrix:

    * `matrix` is a list of lists whose determinant should be calculated
    * If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
    * If `matrix` is not square, raise a `ValueError` with the message `matrix must be a square matrix`
    * The list `[[]]` represents a `0x0` matrix
    * Returns: the determinant of `matrix`

1. Write a function `def minor(matrix):` that calculates the minor matrix of a matrix:

    * `matrix` is a list of lists whose minor matrix should be calculated
    * If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
    * If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
    * Returns: the minor matrix of `matrix`

2. Write a function `def cofactor(matrix):` that calculates the cofactor matrix of a matrix:

    * `matrix` is a list of lists whose cofactor matrix should be calculated
    * If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
    * If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
    * Returns: the cofactor matrix of `matrix`

3. Write a function `def adjugate(matrix):` that calculates the adjugate matrix of a matrix:

    * `matrix` is a list of lists whose adjugate matrix should be calculated
    * If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
    * If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
    * Returns: the adjugate matrix of `matrix`

4. Write a function `def inverse(matrix):` that calculates the inverse of a matrix:

    * `matrix` is a list of lists whose inverse should be calculated
    * If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
    * If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
    * Returns: the inverse of `matrix`, or `None` if `matrix` is singular

5. Write a function `def definiteness(matrix):` that calculates the definiteness of a matrix:

    * `matrix` is a `numpy.ndarray` of shape `(n, n)` whose definiteness should be calculated
    * If `matrix` is not a `numpy.ndarray`, raise a `TypeError` with the message `matrix must be a numpy.ndarray`
    * If `matrix` is not a valid matrix, return `None`
    * Return: the string `Positive definite`, `Positive semi-definite`, `Negative semi-definite`, `Negative definite`, or `Indefinite` if the matrix is positive definite, positive semi-definite, negative semi-definite, negative definite of indefinite, respectively
    * If `matrix` does not fit any of the above categories, return `None`
    * You may `import numpy as np`

