#!/usr/bin/env python3
"""represents a normal distribution:"""


class Normal:
    """normal distribution"""

    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0., stddev=1.):
        """normal distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            variance = [(x - self.mean) ** 2 for x in data]
            self.stddev = (sum(variance) / len(variance)) ** (1 / 2)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return float((x - self.mean)/self.stddev)

    def x_value(self, z):
        """"Calculates the x-value of a given z-score"""
        return float(self.stddev*z + self.mean)

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        n = self.e**(-((x - self.mean)**2)/(2*self.stddev**2))
        den = (self.stddev*(2*self.pi)**0.5)
        return n / den

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        x = (x - self.mean)/(self.stddev*(2)**0.5)
        erf = (2/(self.pi)**0.5)*(x - x**3/3 + x**5/10 - x**7/42 + x**9/216)
        return 0.5*(1 + erf)
