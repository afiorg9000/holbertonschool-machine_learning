#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
# Remove the Weighted_Price column
df = df.drop('Weighted_Price', axis=1)

# Rename the Timestamp column to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.date

# Index the DataFrame on Date
df = df.set_index('Date')

# Fill missing values using forward fill method
df['Close'] = df['Close'].fillna(method='ffill')
df[['High', 'Low', 'Open']] = df[['High', 'Low', 'Open']].fillna(df['Close'].values.reshape(-1, 1))
df[['Volume_(BTC)', 'Volume_(Currency)']] = df[['Volume_(BTC)', 'Volume_(Currency)']].fillna(0)

# Select data from 2017 and beyond at daily intervals
df = df.loc['2017':].resample('D').agg({'High': 'max', 'Low': 'min', 'Open': 'mean', 'Close': 'mean', 'Volume_(BTC)': 'sum', 'Volume_(Currency)': 'sum'})

# Plot the data
df.plot(figsize=(12,8), subplots=True)
plt.show()
