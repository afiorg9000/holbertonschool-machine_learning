#!/usr/bin/env python3
"""Bayesian optimization on a noiseless 1D Gaussian process:"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian optimization on a noiseless 1D Gaussian process:"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """class constructor"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min, max = bounds
        self.X_s = np.linspace(min, max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
