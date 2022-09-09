#!/usr/bin/env python3
"""determines if you should stop gradient descent early:"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """should stop gradient descent early:"""
    if cost < (opt_cost - threshold):
        count = 0
        return False, count

    else:
        count += 1

        if count >= patience:
            return True, count

    return False, count
