#!/usr/bin/env python3
"""Bimodal Distribution"""


class Binomial:
    """Bimodal class"""

    def __init__(self, data=None, n=1, p=0.5):
        """Constructor"""
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            elif p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                mean = sum(data) / len(data)
                variance = 0
                for i in data:
                    variance += (i - mean) ** 2
                variance = variance/len(data)
                q = variance / mean
                self.p = 1 - q
                self.n = round(mean / self.p)
                self.p = mean / self.n

    def pmf(self, k):
        """Probability mass function"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        n_x = (factorial(self.n) / (factorial(k) * factorial(self.n - k)))
        pms = n_x * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return pms

    def cdf(self, k):
        """Cumulative Distribution Function"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf


def factorial(n):
    """Factorial"""
    if n < 0:
        raise ValueError('n must be a positive value')
    elif n == 0:
        return 1
    else:
        return n * factorial(n - 1)
