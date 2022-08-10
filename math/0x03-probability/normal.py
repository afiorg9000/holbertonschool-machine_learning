#!/usr/bin/env python3
"""represents a normal distribution:"""


class Normal:
    """normal distribution"""

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
