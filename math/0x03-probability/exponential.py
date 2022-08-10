#!/usr/bin/env python3
"""represents an exponential distribution:"""


class Exponential:
    """Exponential distriution class"""
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
                self.lambtha = float(len(data) / sum(data)) # Divide the sum by the number of entries.

    def pdf(self, x):
        """probability density function"""
        e = 2.7182818285
        if x < 0:
            return 0
        return (self.lambtha * (e ** (- self.lambtha * x)))

    def cdf(self, x):
        """CDF"""
        e = 2.7182818285
        if x < 0:
            return 0
        return (1 - (e ** (- self.lambtha * x)))
