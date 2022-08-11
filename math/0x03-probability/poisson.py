#!/usr/bin/env python3
"""It estimates how many times an event can happen in a specified time"""


class Poisson:
    """Poisson distriution class"""
    def __init__(self, data=None, lambtha=1.):
        """constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)  # value for average
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """probability mass function"""
        e = 2.7182818285
        if type(k) is not int:
            k = int(k)  # number of occurences
        if k < 0:
            return 0
        return ((self.
                 lambtha ** k) * (e ** (- self.lambtha))) / self.factorial(k)

    def factorial(self, k):
        """factorial of k"""
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
        return factorial

    def cdf(self, k):
        """Cumulative distribution function"""
        e = 2.7182818285
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += ((e ** (- self.
                           lambtha) * (self.lambtha ** i))) / self.factorial(i)
        return cdf
