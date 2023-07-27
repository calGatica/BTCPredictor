import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mpl_dates
import numpy as np

data = pd.read_csv('BTC-USD_max.csv')
ohlc = data.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]

mean = []
for i in range(len(ohlc)):
    mean.append(np.mean(ohlc[['Open', 'High', 'Low', 'Close']].loc[i]))

ohlc['Mean'] = mean
# ohlc.to_csv('btcData.csv', index=False)



plt.style.use('ggplot')

ohlc['Date'] = pd.to_datetime(ohlc['Date'])
ohlc['Date'] = ohlc['Date'].apply(mpl_dates.date2num)
ohlc = ohlc.astype(float)

# Creating Subplots
fig, ax = plt.subplots()

candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

# Setting labels & titles
ax.set_xlabel('Date')
ax.set_ylabel('Price')
fig.suptitle('Daily Candlestick Chart of BTC-USD')

# Formatting Date
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()

fig.tight_layout()

plt.plot(ohlc.Date, ohlc.Mean, '--')

plt.show()

