#!/usr/bin/env python3
"""Bayesian optimization on a noiseless 1D Gaussian process:"""
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """calculates the next best sample location"""
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is True:
            Y_s = np.min(self.gp.Y)
        else:
            Y_s = np.max(self.gp.Y)

        with np.errstate(divide='warn'):
            imp = Y_s - mu - self.xsi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(ei)]
        return X_next, ei

    def optimize(self, iterations=100):
        """Optimizes the black-box function"""
        X_opt, Y_opt = None, None
        for i in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in self.gp.X:
                break
            Y_next = self.f(X_next)[0]
            self.gp.update(X_next, Y_next)
            if self.minimize is True:
                if Y_opt is None or Y_next < Y_opt:
                    X_opt, Y_opt = X_next, Y_next
            else:
                if Y_opt is None or Y_next > Y_opt:
                    X_opt, Y_opt = X_next, Y_next
        return X_opt, Y_opt
