#!/usr/bin/env python3
"""creates a pd.DataFrame from a np.ndarray:"""
import numpy as np
import pandas as pd

def from_numpy(array):
    """creates a pd.DataFrame from a np.ndarray:"""
    columns = [chr(ord('A') + i) for i in range(array.shape[1])] # generate column labels
    df = pd.DataFrame(array, columns=columns) # create DataFrame
    return df
