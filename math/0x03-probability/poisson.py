#!/usr/bin/env python3
"""It estimates how many times an event can happen in a specified time"""


class Poisson:
    """Poisson distriution class"""
    def __init__(self, data=None, lambtha=1.):
        """constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                self.lambtha = float(sum(data) / len(data))
