import pandas as pd
# from scipy.optimize import curve_fit

# data = pd.read_csv('normalized_btcData.csv')
data = pd.read_csv('btcData.csv')


def Plot(data):
    import matplotlib.pyplot as plt
    from mpl_finance import candlestick_ohlc
    import matplotlib.dates as mpl_dates

    plt.style.use('ggplot')

    df = data.loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Mean']]
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].apply(mpl_dates.date2num)
    df = data.astype(float)

    # Creating Subplots
    fig, ax = plt.subplots()

    candlestick_ohlc(ax, df.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

    # Setting labels & titles
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    fig.suptitle('Daily Candlestick Chart of BTC-USD')

    # Formatting Date
    date_format = mpl_dates.DateFormatter('%d-%m-%Y')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    fig.tight_layout()
    plt.plot(df.Date, df.Mean, '--')
    plt.show()
    return

mos = []
mcs = []
for i in range(len(data)-1):
    i+=1
    mo = (data['Open'].loc[i] - data['Open'].loc[i-1])/(data['Date'].loc[i] - data['Date'].loc[i-1]) #pendiente Open
    mc = (data['Close'].loc[i] - data['Close'].loc[i-1])/(data['Date'].loc[i] - data['Date'].loc[i-1]) #pendiente Close
    mos.append(mo)
    mcs.append(mc)

mos.insert(0, -99)
mcs.insert(0, -99)

data['OpenSlope'] = mos
data['CloseSlope'] = mcs
data = data.drop(0)


from sklearn.model_selection import train_test_split
# T, t = tf.keras.utils.split_dataset()
T, t = train_test_split(data, test_size=0.2, shuffle=False)
input_T = T[['Date', 'Open', 'High', 'Low', 'Close', 'OpenSlope', 'CloseSlope']]
input_t = t[['Date', 'Open', 'High', 'Low', 'Close', 'OpenSlope', 'CloseSlope']]
output_T = T[['Date', 'Mean']]
output_t = t[['Date', 'Mean']]

#Shifting mean to predict value of next day
output_T = output_T.shift(-1)
output_t = output_t.shift(-1)

# this left a NaN value at the end of the DF, have to drop it.
input_T = input_T.drop(input_T.tail(1).index)
input_t = input_t.drop(input_t.tail(1).index)
output_T = output_T.drop(output_T.tail(1).index)
output_t = output_t.drop(output_t.tail(1).index)

# input_T, input_t = tf.keras.utils.split_dataset(data, left_size=0.8) #T=Train, t=test


import tensorflow as tf

hide1 = tf.keras.layers.Dense(units = 6, input_shape=[6])
hide2 = tf.keras.layers.Dense(units = 6)
hide3 = tf.keras.layers.Dense(units = 12)
hide4 = tf.keras.layers.Dense(units = 6)
# hide2 = tf.keras.layers.Dense(units = 6)
out = tf.keras.layers.Dense(units = 1)

model = tf.keras.Sequential([hide1, hide2, hide3, hide4, out])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.01),
    loss='mean_squared_error'
    )

print('comenzando entrenamiento')
hist = model.fit(input_T[['Open', 'High', 'Low', 'Close', 'OpenSlope', 'CloseSlope']], output_T['Mean'], epochs=100, verbose=False)


def lossPlot(fittedModel):
    import matplotlib.pyplot as plt
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(fittedModel.history["loss"])
    plt.show()
    return

print('Prediction...')
prediction = model.predict(input_t[['Open', 'High', 'Low', 'Close', 'OpenSlope', 'CloseSlope']])
Pred = pd.DataFrame({'Date': input_t.Date.values, 'prediction': prediction.reshape(len(prediction))})
Pred.to_csv()


def PlotAll(data, pred, A=0.7):
    import matplotlib.pyplot as plt
    from mpl_finance import candlestick_ohlc
    import matplotlib.dates as mpl_dates

    plt.style.use('ggplot')

    df = data.loc[:, ['Date', 'Open', 'High', 'Low', 'Close', 'Mean']]
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].apply(mpl_dates.date2num)
    df = data.astype(float)

    # Creating Subplots
    fig, ax = plt.subplots()

    candlestick_ohlc(ax, df.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

    # Setting labels & titles
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    fig.suptitle('Daily BTC-USD')

    # Formatting Date
    date_format = mpl_dates.DateFormatter('%d-%m-%Y')
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()


    fig.tight_layout()
    plt.plot(df.Date, df.Mean, '--', color='tab:purple', alpha=A, label='True Mean')
    plt.plot(pred.Date, pred.prediction, '.', color='blue', alpha=A, label='Predicted Mean')
    plt.legend(loc=0)
    plt.show()
    return

















