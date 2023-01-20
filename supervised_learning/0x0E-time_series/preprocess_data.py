#!/usr/bin/env python3
"""Preprocess our bitcoin data"""


import numpy as np
import matplotlib.pyplot as plt


def preprocess():
    """
    Preprocess our bitcoin data
    """
    # 1 for no combining, must be a factor of 1440
    # should match param in forecast
    combine = 5
    """
    Skip a lot of empty data at the start of both our data sets.
    Start coinbase at ~1422752580, 2/1/2015
    Start bitstamp at ~1380585600, 10/1/2013
    Column order:
    Timestamp, Open, High, Low, Close, Volume_(BTC),
    Volume_(Currency), Weighted_Price
    """
    file = 'bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
    # Skip a bunch of NaN data and then about the first year where btc
    # volume does not match as well to current time.
    bitstamp = np.genfromtxt(file, delimiter=',', skip_header=1045679)
    print("Initial")
    print(bitstamp[:20])
    """nan entries indicate no changes. There may be some lost data but seems
    small enough to be noise. Set volume to 0 here."""
    bitsnan = np.isnan(bitstamp)[:, 5]
    bitstamp[bitsnan, 5:7] = 0
    print("NaN volume to 0")
    print(bitstamp[:20])

    """
    Replace nan open/close/high/low with previous minute's close.
    """
    fillidxs = np.where(bitsnan, 0, np.arange(bitsnan.shape[0]))
    np.maximum.accumulate(fillidxs, out=fillidxs)
    bitstamp[bitsnan, 4] = bitstamp[fillidxs[bitsnan], 4]
    bitstamp[bitsnan, 1:4] = bitstamp[bitsnan, 4, None]
    bitstamp[bitsnan, 7] = bitstamp[bitsnan, 4]
    print("NaNs replaced")
    print(bitstamp[:20])
    # Exploration uses these.
    np.save("bitstamp_denan.npy", bitstamp)

    """
    There is one time that looks like a data error in bitstamp data. Close
    price goes down to $1.50. We'll take the next minute's open for the
    close price for this time since that looks normal and we're using the
    close price column.
    """
    outlier = bitstamp[:, 4].argmin()
    print("outlier", outlier)
    print(bitstamp[outlier:outlier + 2])
    print(bitstamp[outlier, 0])
    bitstamp[outlier, 4] = bitstamp[outlier + 1, 1]

    """
    Cutting out ~2.5 years of initial data where the BTC volume is much larger
    than current to make data more stationary.
    """
    bitstamp = bitstamp[1000000:]
    data = bitstamp
    dataprices = data[1440 + combine:-60, 3]
    dataratios = data[1500 + combine:, 4] / data[1440 + combine:-60, 4]
    datalabels = data[1500 + combine:, 4]
    datatimes = data[:-1500 - combine, 0]
    np.save("datalabels.npy", datalabels)
    np.save("dataprices.npy", dataprices)
    np.save("dataratios.npy", dataratios)
    np.save("datatimes.npy", datatimes)
    del dataprices, datatimes, datalabels
    """
    If we're going to combine minutes into larger time blocks to reduce
    the input size we need to do it before further processing. Since we're
    about to do a ratio of close to open we only need to move that up by the
    amount of minutes being combined. We do need to sum the volume of BTC as
    well.
    """
    data[1:, 1] = data[:-1, 4]
    data = data[1:]
    if combine > 1:
        data[:-combine + 1, 4] = data[combine - 1:, 4]
        btcvolsum = np.zeros((data.shape[0]))
        print(btcvolsum.shape)
        for i in range(combine):
            btcvolsum[:-i or None] += data[i:, 5]
        print("btcvolsum", btcvolsum)
        print(btcvolsum.shape)
        print(data.shape)
        data[:-combine + 1, 5] = btcvolsum[:-combine + 1]
    data = data[:-1500 - combine + 1]

    """
    Add a somewhat standardizied open/close ratio and make it column 8
    then remove unneeded price information. This should prevent the absolute
    price fluctuating over time messing up our distribution. Do this with
    split datasets before we remove price to help memory concerns.
    """
    stdratio = data[:, 4] / data[:, 1]
    data = np.append(data[:], stdratio[:, None], axis=1)
    data = data[:, [8, 5]].copy()  # copy to reduce memory needs

    assert (data.shape[0] == dataratios.shape[0])
    print("Added ratios")

    """
    Remaining data is open/close ratio and btc volume.

    We need a 24 hour window of time, the price at the end of that window,
    the price ratio between our last point and the prediction point,
    and the price an hour after that window. These are in data, dataratios,
    dataprices, and datalabels respectively.
    """
    # Logmax BTC volume
    data[:, 1] = np.log(data[:, 1] + 1)
    data[:, 1] = data[:, 1] / data[:, 1].max()
    np.save("data.npy", data)


if __name__ == "__main__":
    preprocess()