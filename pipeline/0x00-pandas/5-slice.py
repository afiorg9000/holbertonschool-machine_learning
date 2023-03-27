#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Slice DataFrame along columns High, Low, Close, and Volume_BTC, taking every 60th row
df = df[['High', 'Low', 'Close', 'Volume_(BTC)']][::60]


print(df.tail())
