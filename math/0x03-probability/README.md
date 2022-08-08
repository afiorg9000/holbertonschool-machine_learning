# 0x03. Probability


## Learning Objectives

-   What is probability?
-   Basic probability notation
-   What is independence? What is disjoint?
-   What is a union? intersection?
-   What are the general addition and multiplication rules?
-   What is a probability distribution?
-   What is a probability distribution function? probability mass function?
-   What is a cumulative distribution function?
-   What is a percentile?
-   What is mean, standard deviation, and variance?
-   Common probability distributions

## Tasks


### 0. Initialize Poisson

Create a class `Poisson` that represents a poisson distribution

### 1. Poisson PMF

Update the class `Poisson`

-   Instance method  `def pmf(self, k):`
    -   Calculates the value of the PMF for a given number of “successes”
    -   `k`  is the number of “successes”
        -   If  `k`  is not an integer, convert it to an integer
        -   If  `k`  is out of range, return  `0`
    -   Returns the PMF value for  `k`

### 2. Poisson CDF

Update the class `Poisson`:
-   Instance method  `def cdf(self, k):`
    -   Calculates the value of the CDF for a given number of “successes”
    -   `k`  is the number of “successes”
        -   If  `k`  is not an integer, convert it to an integer
        -   If  `k`  is out of range, return  `0`
    -   Returns the CDF value for  `k`


### 3. Initialize Exponential

Create a class `Exponential` that represents an exponential distribution:

### 4. Exponential PDF

Update the class  `Exponential`:

-   Instance method  `def pdf(self, x):`
    -   Calculates the value of the PDF for a given time period
    -   `x`  is the time period
    -   Returns the PDF value for  `x`
    -   If  `x`  is out of range, return  `0`

### 5. Exponential CDF

Update the class  `Exponential`:

-   Instance method  `def cdf(self, x):`
    -   Calculates the value of the CDF for a given time period
    -   `x`  is the time period
    -   Returns the CDF value for  `x`
    -   If  `x`  is out of range, return  `0`

### 6. Initialize Normal
Create a class `Normal` that represents a normal distribution:

### 7. Normalize Normal

Update the class  `Normal`:

-   Instance method  `def z_score(self, x):`
    -   Calculates the z-score of a given x-value
    -   `x`  is the x-value
    -   Returns the z-score of  `x`
-   Instance method  `def x_value(self, z):`
    -   Calculates the x-value of a given z-score
    -   `z`  is the z-score
    -   Returns the x-value of  `z`


### 8. Normal PDF

Update the class  `Normal`:

-   Instance method  `def pdf(self, x):`
    -   Calculates the value of the PDF for a given x-value
    -   `x`  is the x-value
    -   Returns the PDF value for  `x`


### 9. Normal CDF

Update the class  `Normal`:

-   Instance method  `def cdf(self, x):`
    -   Calculates the value of the CDF for a given x-value
    -   `x`  is the x-value
    -   Returns the CDF value for  `x`


### 10. Initialize Binomial

Create a class `Binomial` that represents a binomial distribution


### 11. Binomial PMF
Update the class  `Binomial`:

-   Instance method  `def pmf(self, k):`
    -   Calculates the value of the PMF for a given number of “successes”
    -   `k`  is the number of “successes”
        -   If  `k`  is not an integer, convert it to an integer
        -   If  `k`  is out of range, return  `0`
    -   Returns the PMF value for  `k`


### 12. Binomial CDF

Update the class  `Binomial`:

-   Instance method  `def cdf(self, k):`
    -   Calculates the value of the CDF for a given number of “successes”
    -   `k`  is the number of “successes”
        -   If  `k`  is not an integer, convert it to an integer
        -   If  `k`  is out of range, return  `0`
    -   Returns the CDF value for  `k`
    -   _Hint: use the  `pmf`  method_
