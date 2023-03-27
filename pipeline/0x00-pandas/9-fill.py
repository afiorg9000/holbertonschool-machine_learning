#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove "Weighted_Price" column
df = df.drop(columns=['Weighted_Price'])

# Fill missing data points
df[['Close', 'High', 'Low', 'Open']] = df[['Close', 'High', 'Low', 'Open']].fillna(method='ffill')
df[['Volume_(BTC)', 'Volume_(Currency)']] = df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)


print(df.head())
print(df.tail())
