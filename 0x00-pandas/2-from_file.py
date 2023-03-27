#!/usr/bin/env python3
"""loads data from a file as a pd.DataFrame:"""
import pandas as pd

def from_file(filename, delimiter):
    """loads data from a file as a pd.DataFrame:"""
    df = pd.read_csv(filename, delimiter=delimiter) # load data from file
    return df
