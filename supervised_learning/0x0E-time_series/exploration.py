#!/usr/bin/env python3


import numpy as np

import matplotlib
from matplotlib import pyplot
plt = pyplot

matplotlib.rc('font', size=12)
matplotlib.rcParams['lines.linewidth'] = 1.0

np.set_printoptions(linewidth=140, precision=5)
bitstamp = np.load("bitstamp_denan.npy")
coinbase = np.load("coinbase_denan.npy")

"""
Random looking close price outlier ($1.50), could not find a related
news event. Likely a recording error of some sort. We'll set close
price to next day's open which is not the same and looks more reasonable.
"""

outlier = bitstamp[:, 4].argmin()
print("outlier", outlier)
print(bitstamp[outlier:outlier + 2])
print(bitstamp[outlier, 0])
bitstamp[outlier, 4] = bitstamp[outlier + 1, 1]

print("Check btc volume outliers")
print(bitstamp[bitstamp[:, 5].argsort()[-40:][::-1],5 ])
bitstamp = bitstamp[1000000:]
"""Take open/close ratio, then subtract 1 and inverse any numbers less than 1
to normalize doubling/halving of prices to same scale"""
normratio = bitstamp[:, 4] / bitstamp[:, 1]
print(normratio[:20])
#normratio[normratio < 1] = -1 / normratio[normratio < 1] + 1
print(normratio[:20])
#normratio[normratio >= 1] = normratio[normratio >= 1] - 1
print(normratio[:20])
bitstamp = np.append(bitstamp, normratio[:, None], axis=1)
# will drift about 20 seconds every day when accounting for leap year
# 86459.178 minutes per day average
normday = np.mod(bitstamp[:, 0], 86459)
normday = 2 * np.pi * normday / 86459
bitstamp = np.append(bitstamp, np.sin(normday)[:, None], axis=1)
del normday
normyear = np.mod(bitstamp[:, 0, None], 31557600)
normyear = 2 * np.pi * normyear / 31557600
bitstamp = np.append(bitstamp, np.sin(normyear)[:, 0, None], axis=1)
del normyear

#ratio graphs
years = 5
if 0:
    plt.plot(bitstamp[:, 0], bitstamp[:, 8])
    plt.ylabel("End / Start ratio")
    plt.xticks([bitstamp[0, 0] + x * 31557600 for x in range(years)])
    plt.xlabel("Timestamp")
    plt.show()
    stds = np.std(bitstamp[:, 8], axis=0, keepdims=True)
    means = np.mean(bitstamp[:, 8], axis=0, keepdims=True)
    normed = .5 * np.tanh(.01 * (bitstamp[:, 8] - means) / stds)
    plt.plot(bitstamp[:, 0], normed)
    plt.ylabel("quasiratio tanh normed")
    plt.xticks([bitstamp[0, 0] + x * 31557600 for x in range(years)])
    plt.xlabel("Timestamp")
    plt.show()
#btc vol graphs
if 1:
    plt.plot(bitstamp[:, 0], bitstamp[:, 5])
    plt.ylabel("BTC Volume")
    plt.xlabel("Timestamp")
    plt.xticks([bitstamp[0, 0] + x * 31557600 for x in range(years)])
    plt.show()
    """
    stds = np.std(bitstamp[:, 5], axis=0, keepdims=True)
    means = np.mean(bitstamp[:, 5], axis=0, keepdims=True)
    normed = .5 * np.tanh(.01 * (bitstamp[:, 5] - means) / stds)
    plt.plot(bitstamp[:, 0], normed)
    plt.ylabel("BTC Volume")
    plt.xlabel("Timestamp")
    plt.xticks([bitstamp[0, 0] + x * 31557600 for x in range(years)])
    plt.show()
    """
    logvol = np.log(bitstamp[:, 5] + 1)
    plt.plot(bitstamp[:, 0], logvol)
    plt.ylabel("BTC Log Volume")
    plt.xlabel("Timestamp")
    plt.xticks([bitstamp[0, 0] + x * 31557600 for x in range(years)])
    plt.show()
    """
    stds = np.std(logvol)
    means = np.mean(logvol)
    normed = .5 * np.tanh(.01 * (logvol - means) / stds)
    plt.plot(bitstamp[:, 0], normed)
    plt.ylabel("BTC Log Volume")
    plt.xlabel("Timestamp")
    plt.xticks([bitstamp[0, 0] + x * 31557600 for x in range(years)])
    plt.show()
    """

#price graphs
if 0:
    plt.plot(bitstamp[:, 0], bitstamp[:, 4])
    plt.ylabel("Close Price")
    plt.xlabel("Timestamp")
    plt.xticks([bitstamp[0, 0] + x * 31557600 for x in range(years)])
    plt.show()
    halfsize = int(bitstamp.shape[0] * 1 / 3)
    plt.plot(bitstamp[:halfsize, 0], bitstamp[:halfsize, 4])
    plt.ylabel("close price")
    plt.xlabel("Timestamp")
    plt.xticks([bitstamp[0, 0] + x * 31557600 for x in range(int(years / 2))])
    plt.show()
    plt.plot(bitstamp[:, 0], bitstamp[:, 4] - bitstamp[:, 1])
    plt.ylabel("close - open price")
    plt.xlabel("Timestamp")
    plt.xticks([bitstamp[0, 0] + x * 31557600 for x in range(years)])
    plt.show()
    plt.plot(bitstamp[:, 0], bitstamp[:, 4])
    plt.ylabel("close price")
    plt.xticks([bitstamp[0, 0] + x * 31557600 for x in range(years)])
    plt.show()

exit()

#change ratio, btc volume, daysin, yearsin
print(bitstamp.shape)
print("ratio")
corrcoef = np.corrcoef(bitstamp.T)
print(corrcoef[1:8, 1:8])
print("      ""  Open  ""  High  ""  Low   "" Close  "" BTCVol "" USDVOL ""  Avg   ")
print("Open  {}".format(corrcoef[1, 1:8]))
print("High  {}".format(corrcoef[2, 1:8]))
print("Low   {}".format(corrcoef[3, 1:8]))
print("Close {}".format(corrcoef[4, 1:8]))
print("BTCVol{}".format(corrcoef[5, 1:8]))
print("USDVol{}".format(corrcoef[6, 1:8]))
print("Avg   {}".format(corrcoef[7, 1:8]))
exit()
bitstamp = bitstamp[:, [8, 5, 9, 10]].copy()
print("ratio, btc vol, daysin, yearsin")
print(np.corrcoef(bitstamp.T))


"""Take open/close ratio, then subtract 1 and inverse any numbers less than 1
to normalize doubling/halving of prices to same scale"""
normratio = coinbase[:, 4] / coinbase[:, 1]
print(normratio[:20])
normratio[normratio < 1] = -1 / normratio[normratio < 1] + 1
print(normratio[:20])
normratio[normratio >= 1] = normratio[normratio >= 1] - 1
print(normratio[:20])
coinbase = np.append(coinbase, normratio[:, None], axis=1)
print("open, high, low, close, btcvol, usdvol, trade avg, close:open")
print(np.corrcoef(coinbase.T))


print(coinbase[0])

print("Check similarity between two data sets")
timesync = np.where(bitstamp[:, 0] == coinbase[0][0])[0][0]
print(timesync)
print(coinbase[:50])
print(bitstamp[timesync:timesync + 50])