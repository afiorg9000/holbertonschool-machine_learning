# 0x02 Calculus

> Calculus is an important field in mathematics and it plays an integral role in many machine learning algorithms. If you want to understand what’s going on under the hood in your machine learning work as a data scientist, you’ll need to have a solid grasp of the fundamentals of calculus. This project covers the first part of calculus that covers concepts for algorithms such as the gradient descent algorithm and backpropagation to train deep learning neural networks implementation with Python for machine learning.

At the end of this project I was able to solve these conceptual questions:

* Summation and Product notation
* What is a series?
* Common series
* What is a derivative?
* What is the product rule?
* What is the chain rule?
* Common derivative rules
* What is a partial derivative?
* What is an indefinite integral?
* What is a definite integral?
* What is a double integral?

## Tasks

0. Type the number of the correct answer in your answer file

    ![Task 0](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task0.png)

1. Type the number of the correct answer in your answer file

    ![Task 1](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task1.png)

2. Type the number of the correct answer in your answer file

    ![Task 2](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task2.png)

3. Type the number of the correct answer in your answer file

    ![Task 3](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task3.png)

4. Type the number of the correct answer in your answer file

    ![Task 4](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task4.png)

5. Type the number of the correct answer in your answer file

    ![Task 5](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task5.png)

6. Type the number of the correct answer in your answer file

    ![Task 6](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task6.png)

7. Type the number of the correct answer in your answer file

    ![Task 7](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task7.png)

8. Type the number of the correct answer in your answer file

    ![Task 8](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task8.png)

9. Write a function `def summation_i_squared(n):` that calculates:

    ![Task 9](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task9.png)

    * `n` is the stopping condition
    * Return the integer value of the sum
    * If `n` is not a valid number, return `None`
    * You are not allowed to use any loops

10. Write a function `def poly_derivative(poly):` that calculates the derivative of a polynomial:

    * `poly` is a list of coefficients representing a polynomial
        * the index of the list represents the power of `x` that the coefficient belongs to
        * Example: if f(x) = x^3 + 3x +5, `poly` is equal to `[5, 3, 0, 1]`
    * If `poly` is not valid, return `None`
    * If the derivative is `0`, return `[0]`
    * Return a new list of coefficients representing the derivative of the polynomial

11. Type the number of the correct answer in your answer file

    ![Task 11](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task11.png)

12. Type the number of the correct answer in your answer file

    ![Task 12](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task12.png)

13. Type the number of the correct answer in your answer file

    ![Task 13](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task13.png)

14. Type the number of the correct answer in your answer file

    ![Task 14](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task14.png)

15. Type the number of the correct answer in your answer file

    ![Task 15](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task15.png)

16. Type the number of the correct answer in your answer file

    ![Task 16](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x02-calculus/img/task16.png)

17. Write a function `def poly_integral(poly, C=0):` that calculates the integral of a polynomial:

    * `poly` is a list of coefficients representing a polynomial
        * the index of the list represents the power of `x` that the coefficient belongs to
        * Example: if f(x) = x^3 + 3x +5, `poly` is equal to `[5, 3, 0, 1]`
    * `C` is an integer representing the integration constant
    * If a coefficient is a whole number, it should be represented as an integer
    * If `poly` or `C` are not valid, return `None`
    * Return a new list of coefficients representing the integral of the polynomial
    * The returned list should be as small as possible
