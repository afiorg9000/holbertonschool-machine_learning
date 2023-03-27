#!/usr/bin/env python3
"""computes to policy with a weight of a matrix."""
import numpy as np

def policy(matrix, weight):
    """computes to policy with a weight of a matrix."""
    # Compute dot product of the matrix and weight
    dot_product = np.dot(matrix, weight)
    
    # Compute softmax of the dot product
    softmax = np.exp(dot_product) / np.sum(np.exp(dot_product))
    
    return softmax

def policy_gradient(state, weight):
    # Compute the policy and its gradient
    p = policy(state, weight)
    action = np.random.choice(range(p.shape[1]), p=p.ravel())
    grad_log = state.T - np.dot(state.T, p)
    
    return action, grad_log