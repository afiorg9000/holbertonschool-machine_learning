
# 0x00. Linear Algebra

By Alexa Orrico, Software Engineer at Holberton School

![enter image description here](https://bestdelegate.com/wp-content/uploads/2015/11/morpheus.jpg)

Resources
---------

**Read or watch**:

-   [Introduction to vectors](https://intranet.hbtn.io/rltoken/C05mTOfKzZgz_AVSosNKIw)
-   [What is a matrix? (not the matrix)](https://intranet.hbtn.io/rltoken/vLe4BBPfmLXy2s_Idqo87w)
-   [Transpose](https://intranet.hbtn.io/rltoken/xHWwQjqH9tgEcskvFQaV7A)
-   [Understanding the dot product](https://intranet.hbtn.io/rltoken/2tYcOFY35stXjd0nhTpgFA)
-   [Matrix Multiplication](https://intranet.hbtn.io/rltoken/pV4znghCxaXAAny4Ou-cNw)
-   [What is the relationship between matrix multiplication and the dot product?](https://intranet.hbtn.io/rltoken/ih50DhE4FvilyItYPo1x5A)
-   [The Dot Product, Matrix Multiplication, and the Magic of Orthogonal Matrices (advanced)](https://intranet.hbtn.io/rltoken/DnAvjbmojZutluWV9OJVOg)
-   [numpy tutorial (until Shape Manipulation (excluded))](https://intranet.hbtn.io/rltoken/MBHHb0eiN0OummbEdI9g_Q)
-   [numpy basics (until Universal Functions (included))](https://intranet.hbtn.io/rltoken/L8RdIDGi3GGO-_erGcMORg)
-   [array indexing](https://intranet.hbtn.io/rltoken/1LPk4EosRetS_C7eX-mQNA)
-   [numerical operations on arrays](https://intranet.hbtn.io/rltoken/slRzAgt6aom5-Nj5XSdUcQ)
-   [Broadcasting](https://intranet.hbtn.io/rltoken/xgq6QIOHufhg8lHCZn0jwA)
-   [numpy mutations and broadcasting](https://intranet.hbtn.io/rltoken/Q5FEVV4BArJtnJnbReng7Q)

**References**:

-   [numpy.ndarray](https://intranet.hbtn.io/rltoken/Ah-QtZhAhFSYnloj837a8Q)
-   [numpy.ndarray.shape](https://intranet.hbtn.io/rltoken/mvx-STJbJ4Nn1N_BFfpnaQ)
-   [numpy.transpose](https://intranet.hbtn.io/rltoken/I1V8iDWar7Hnoh_VwQzZ_Q)
-   [numpy.ndarray.transpose](https://intranet.hbtn.io/rltoken/iv73fN04gTbpeV_XcIIaPQ)
-   [numpy.matmul](https://intranet.hbtn.io/rltoken/MbHJEqjwavimnL8HRtaYCA)


Learning Objectives
-------------------

At the end of this project, you are expected to be able to [explain to anyone](https://alx-intranet.hbtn.io/rltoken/mGnreK2ar-4GUXzcb9OtXw "explain to anyone"), **without the help of Google**:

### General

-   What is a vector?
-   What is a matrix?
-   What is a transpose?
-   What is the shape of a matrix?
-   What is an axis?
-   What is a slice?
-   How do you slice a vector/matrix?
-   What are element-wise operations?
-   How do you concatenate vectors/matrices?
-   What is the dot product?
-   What is matrix multiplication?
-   What is  `Numpy`?
-   What is parallelization and why is it important?
-   What is broadcasting?

Requirements
------------

### General

-   Allowed editors: `vi`, `vim`, `emacs`
-   All your files will be interpreted/compiled on Ubuntu 20.04 LTS using `python3` (version 3.8)
-   Your files will be executed with `numpy` (version 1.19.2)
-   All your files should end with a new line
-   The first line of all your files should be exactly `#!/usr/bin/env python3`
-   A `README.md` file, at the root of the folder of the project, is mandatory
-   Your code should follow `pycodestyle` (version 2.6)
-   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
-   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
-   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'`and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
-   Unless otherwise noted, you are not allowed to import any module
-   All your files must be executable
-   The length of your files will be tested using `wc`

More Info
---------
### Installing Ubuntu 20.04 and Python 3.8
Follow the instructions listed in `Using Vagrant on your personal computer`, should be using `ubuntu/focal64`.

Python 3.8 comes pre-installed on Ubuntu 20.04. How convenient! You can confirm this with `python3 -V`

### Installing pip (latest)
[pip installation](https://intranet.hbtn.io/rltoken/bnipr2zxol-aSqNNCglaFg)

### Installing numpy 1.19.2, scipy 1.6.2, and pycodestyle 2.6
```
$ pip install --user numpy==1.19.2
$ pip install --user scipy==1.6.2
$ pip install --user pycodestyle==2.6
```
To check that all have been successfully downloaded, use `pip list`.

## Tasks

### 0. Slice Me Up

Complete the following source code (found below):

-   `arr1`  should be the first two numbers of  `arr`
-   `arr2`  should be the last five numbers of  `arr`
-   `arr3`  should be the 2nd through 6th numbers of  `arr`
-   You are not allowed to use any loops or conditional statements
-   Your program should be exactly 8 lines

```
alexa@ubuntu-focal:0x00-linear_algebra$ cat 0-slice_me_up.py 
#!/usr/bin/env python3
arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]
arr1 =  # your code here
arr2 =  # your code here
arr3 =  # your code here
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))
alexa@ubuntu-focal:0x00-linear_algebra$ ./0-slice_me_up.py 
The first two numbers of the array are: [9, 8]
The last five numbers of the array are: [9, 4, 1, 0, 3]
The 2nd through 6th numbers of the array are: [8, 2, 3, 9, 4]
alexa@ubuntu-focal:0x00-linear_algebra$ wc -l 0-slice_me_up.py 
8 0-slice_me_up.py
alexa@ubuntu-focal:0x00-linear_algebra$ 
```
**Repo:**

-   GitHub repository: `holbertonschool-machine_learning`
-   Directory: `math/0x00-linear_algebra`
-   File: `0-slice_me_up.py`

### 1. Trim Me Down
Complete the following source code (found below):

* `the_middle` should be a 2D matrix containing the 3rd and 4th columns of `matrix`
* You are not allowed to use any conditional statements
* You are only allowed to use one `for` loop
* Your program should be exactly 6 lines

```
alexa@ubuntu-focal:0x00-linear_algebra$ cat 1-trim_me_down.py 
#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
# your code here
print("The middle columns of the matrix are: {}".format(the_middle))
alexa@ubuntu-focal:0x00-linear_algebra$ ./1-trim_me_down.py 
The middle columns of the matrix are: [[9, 4], [7, 3], [4, 6]]
alexa@ubuntu-focal:0x00-linear_algebra$ wc -l 1-trim_me_down.py 
6 1-trim_me_down.py
alexa@ubuntu-focal:0x00-linear_algebra$ 
```
**Repo:**

-   GitHub repository: `holbertonschool-machine_learning`
-   Directory: `math/0x00-linear_algebra`
-   File: `1-trim_me_down.py`


### 2. Size Me Please
Write a function `def matrix_shape(matrix):` that calculates the shape of a matrix:

* You can assume all elements in the same dimension are of the same type/shape
* The shape should be returned as a list of integers

```
alexa@ubuntu-focal:0x00-linear_algebra$ cat 2-main.py 
#!/usr/bin/env python3

matrix_shape = __import__('2-size_me_please').matrix_shape

mat1 = [[1, 2], [3, 4]]
print(matrix_shape(mat1))
mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
print(matrix_shape(mat2))
alexa@ubuntu-focal:0x00-linear_algebra$ ./2-main.py 
[2, 2]
[2, 3, 5]
alexa@ubuntu-focal:0x00-linear_algebra$ 
```
**Repo:**

-   GitHub repository: `holbertonschool-machine_learning`
-   Directory: `math/0x00-linear_algebra`
-   File: `2-size_me_please.py`


### 3. Flip Me Over
Write a function `def matrix_transpose(matrix):` that returns the transpose of a 2D matrix, matrix:

* You must return a new matrix
* You can assume `that matrix` is never empty
* You can assume all elements in the same dimension are of the same type/shape

### 4. Line Up
Write a function `def add_arrays(arr1, arr2):` that adds two arrays element-wise:

* You can assume that `arr1` and `arr2` are lists of ints/floats
* You must return a new list
* If `arr1` and `arr2` are not the same shape, return `None`

### 5. Across The Planes
Write a function `def add_matrices2D(mat1, mat2):` that adds two matrices element-wise:

* You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
* You can assume all elements in the same dimension are of the same type/shape
* You must return a new matrix
* If `mat1` and `mat2` are not the same shape, return `None`

### 6. Howdy Partner
Write a function `def cat_arrays(arr1, arr2):` that concatenates two arrays:

* You can assume that `arr1` and `arr2` are lists of ints/floats
* You must return a new list

### 7. Gettin’ Cozy
Write a function `def cat_matrices2D(mat1, mat2, axis=0):` that concatenates two matrices along a specific axis:

* You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
* You can assume all elements in the same dimension are of the same type/shape
* You must return a new matrix
* If the two matrices cannot be concatenated, return `None`

### 8. Ridin’ Barebac
Write a function `def mat_mul(mat1, mat2):` that performs matrix multiplication:

* You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
* You can assume all elements in the same dimension are of the same type/shape
* You must return a new matrix
* If the two matrices cannot be multiplied, return `None`

### 9. Let The Butcher Slice It
Complete the following source code (found below):

* `mat1` should be the middle two rows of `matrix`
* `mat2` should be the middle two columns of `matrix`
* `mat3` should be the bottom-right, square, 3x3 matrix of `matrix`
* You are not allowed to use any loops or conditional statements
* Your program should be exactly 10 lines

### 10. I’ll Use My Scale
Write a function `def np_shape(matrix):` that calculates the shape of a numpy.ndarray:

* You are not allowed to use any loops or conditional statements
* You are not allowed to use `try/except` statements
* The shape should be returned as a tuple of integers

### 11. The Western Exchange
Write a function `def np_transpose(matrix):` that transposes matrix:

* You can assume that `matrix` can be interpreted as a `numpy.ndarray`
* You are not allowed to use any loops or conditional statements
* You must return a new `numpy.ndarray`

### 12. Bracing The Elements
Write a function `def np_elementwise(mat1, mat2):` that performs element-wise addition, subtraction, multiplication, and division:

* You can assume that `mat1` and `mat2` can be interpreted as `numpy.ndarrays`
* You should return a tuple containing the element-wise sum, difference, product, and quotient, respectively
* You are not allowed to use any loops or conditional statements
* You can assume that `mat1` and `mat2` are never empty

### 13. Cat's Got Your Tongue
Write a function `def np_cat(mat1, mat2, axis=0)` that concatenates two matrices along a specific axis:

* You can assume that `mat1` and `mat2` can be interpreted as `numpy.ndarray`s
* You must return a new `numpy.ndarray`
* You are not allowed to use any loops or conditional statements
* You may use: `import numpy as np`
* You can assume that `mat1` and `mat2` are never empty

### 14. Saddle Up
Write a function `def np_matmul(mat1, mat2):` that performs matrix multiplication:

* You can assume that mat1 and mat2 are numpy.ndarrays
* You are not allowed to use any loops or conditional statements
* You may use: import numpy as np
* You can assume that mat1 and mat2 are never empty
