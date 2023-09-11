# 0x03 Probability

> Probability is the bedrock of Machine Learning, classification models must predict a probability of class membership, algorithms are designed using probability (e.g. Naive Bayes), algorithms are trained under probability frameworks (e.g. maximum likelihood).

At the end of this project I was able to solve these conceptual questions:

* What is probability?
* Basic probability notation
* What is independence? What is disjoint?
* What is a union? intersection?
* What are the general addition and multiplication rules?
* What is a probability distribution?
* What is a probability distribution function? probability mass function?
* What is a cumulative distribution function?
* What is a percentile?
* What is mean, standard deviation, and variance?
* Common probability distributions

## Mathematical Approximations

For the following tasks, I had to use various irrational numbers and functions. Since I´m not able to import any libraries, I used the following approximations:

* π = 3.1415926536
* e = 2.7182818285
* ![erf(x)](https://raw.githubusercontent.com/jhonaRiver/holbertonschool-machine_learning/master/math/0x03-probability/img/math_approx.png)

## Tasks

0. Create a class `Poisson` that represents a poisson distribution:

    * Class contructor `def __init__(self, data=None, lambtha=1.):`
        * `data` is a list of the data to be used to estimate the distribution
        * `lambtha` is the expected number of occurences in a given time frame
        * Sets the instance attribute `lambtha`
            * Saves `lambtha` as a float
        * If `data` is not given, (i.e. `None` (be careful: `not data` has not the same result as `data is None`)):
            * Use the given `lambtha`
            * If `lambtha` is not a positive value or equals to 0, raise a `ValueError` with the message `lambtha must be a positive value`
        * If `data` is given:
            * Calculate the `lambtha` of `data`
            * If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
            * If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`

1. Update the class `Poisson`:

    * Instance method `def pmf(self, k):`
        * Calculates the value of the PMF for a given number of “successes”
        * `k` is the number of “successes”
            * If `k` is not an integer, convert it to an integer
            * If `k` is out of range, return `0`
        * Returns the PMF value for `k`

2. Update the class `Poisson`:

    * Instance method `def cdf(self, k):`
        * Calculates the value of the CDF for a given number of “successes”
        * `k` is the number of “successes”
            * If `k` is not an integer, convert it to an integer
            * If `k` is out of range, return `0`
        * Returns the CDF value for `k`

3. Create a class `Exponential` that represents an exponential distribution:

    * Class contructor `def __init__(self, data=None, lambtha=1.):`
        * `data` is a list of the data to be used to estimate the distribution
        * `lambtha` is the expected number of occurences in a given time frame
        * Sets the instance attribute `lambtha`
            * Saves `lambtha` as a float
        * If `data` is not given (i.e. `None`):
            * Use the given `lambtha`
            * If `lambtha` is not a positive value, raise a `ValueError` with the message `lambtha must be a positive value`
        * If `data` is given:
            * Calculate the `lambtha` of `data`
            * If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
            * If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`

4. Update the class `Exponential`:

    * Instance method `def pdf(self, x):`
        * Calculates the value of the PDF for a given time period
        * `x` is the time period
        * Returns the PDF value for `x`
        * If `x` is out of range, return `0`

5. Update the class `Exponential`:

    * Instance method `def cdf(self, x):`
        * Calculates the value of the CDF for a given time period
        * `x` is the time period
        * Returns the CDF value for `x`
        * If `x` is out of range, return `0`

6. Create a class `Normal` that represents a normal distribution:

    * Class contructor `def __init__(self, data=None, mean=0., stddev=1.):`
        * `data` is a list of the data to be used to estimate the distribution
        * `mean` is the mean of the distribution
        * `stddev` is the standard deviation of the distribution
        * Sets the instance attributes `mean` and `stddev`
            * Saves `mean` and `stddev` as floats
        * If `data` is not given (i.e. `None` (be careful: `not data` has not the same result as `data is None`))
            * Use the given `mean` and `stddev`
            * If `stddev` is not a positive value or equals to 0, raise a `ValueError` with the message `stddev must be a positive value`
        * If `data` is given:
            * Calculate the mean and standard deviation of `data`
            * If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
            * If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`

7. Update the class `Normal`:

    * Instance method `def z_score(self, x):`
        * Calculates the z-score of a given x-value
        * `x` is the x-value
        * Returns the z-score of `x`
    * Instance method `def x_value(self, z):`
        * Calculates the x-value of a given z-score
        * `z` is the z-score
        * Returns the x-value of `z`

8. Update the class `Normal`:

    * Instance method `def pdf(self, x):`
        * Calculates the value of the PDF for a given x-value
        * `x` is the x-value
        * Returns the PDF value for `x`

9. Update the class `Normal`:

    * Instance method `def cdf(self, x):`
        * Calculates the value of the CDF for a given x-value
        * `x` is the x-value
        * Returns the CDF value for `x`

10. Create a class `Binomial` that represents a binomial distribution:

    * Class contructor `def __init__(self, data=None, n=1, p=0.5):`
        * `data` is a list of the data to be used to estimate the distribution
        * `n` is the number of Bernoulli trials
        * `p` is the probability of a “success”
        * Sets the instance attributes `n` and `p`
            * Saves `n` as an integer and `p` as a float
        * If `data` is not given (i.e. `None`)
            * Use the given `n` and `p`
            * If `n` is not a positive value, raise a `ValueError` with the message `n must be a positive value`
            * If `p` is not a valid probability, raise a `ValueError` with the message `p must be greater than 0 and less than 1`
        * If `data` is given:
            * Calculate `n` and `p` from `data`
            * Round `n` to the nearest integer (**rounded, not casting!** The difference is important: `int(3.7)` is not the same as `round(3.7)`)
            * *Hint: Calculate p first and then calculate n. Then recalculate p. Think about why you would want to do it this way?*
            * If `data` is not a `list`, raise a `TypeError` with the message `data must be a list`
            * If `data` does not contain at least two data points, raise a `ValueError` with the message `data must contain multiple values`

11. Update the class `Binomial`:

    * Instance method `def pmf(self, k):`
        * Calculates the value of the PMF for a given number of “successes”
        * `k` is the number of “successes”
            * If `k` is not an integer, convert it to an integer
            * If `k` is out of range, return `0`
        * Returns the PMF value for `k`

12. Update the class `Binomial`:

    * Instance method `def cdf(self, k):`
        * Calculates the value of the CDF for a given number of “successes”
        * `k` is the number of “successes”
            * If `k` is not an integer, convert it to an integer
            * If `k` is out of range, return `0`
        * Returns the CDF value for `k`
        * *Hint: use the `pmf` method*
