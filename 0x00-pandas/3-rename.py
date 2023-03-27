#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Rename column and convert to datetime
df = df.rename(columns={'Timestamp': 'Datetime'})
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

# Display Datetime and Close columns only
df = df[['Datetime', 'Close']]

print(df.tail())
