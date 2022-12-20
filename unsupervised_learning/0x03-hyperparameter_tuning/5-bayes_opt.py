#!/usr/bin/env python3
"""Bayesian optimization on a noiseless 1D Gaussian process:"""
import numpy as np
from numpy import argmax
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
        mu, sig = self.gp.predict(X_s=self.X_s)

        if self.minimize:
            fx_p = np.min(self.gp.Y)
            num = fx_p - mu - self.xsi
        else:
            fx_p = np.max(self.gp.Y)
            num = mu - fx_p - self.xsi

        Z = np.where(sig == 0, 0, num / sig)
        EI = np.where(sig == 0, 0, num * norm.cdf(Z) + sig * norm.pdf(Z))
        EI = np.maximum(EI, 0)
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """Optimizes the black-box function"""
        for i in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in self.gp.X:
                break
            else:
                Y_next = self.f(X_next)
                self.gp.update(X_next, Y_next)
        if self.minimize:
            opt_i = np.argmin(self.gp.Y)
        else:
            opt_i = np.argmax(self.gp.Y)
        return self.gp.X[opt_i], self.gp.Y[opt_i]
