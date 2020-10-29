from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib

dataset_ex_df = pd.read_csv('transaction_data.tsv', sep='\t', header=0)
companyList = dataset_ex_df['TICKER'].unique().tolist()


def Arima(companyName):
    data = dataset_ex_df[dataset_ex_df.TICKER == companyName]
    train_data = data['PRC']
    return train_data


def draw_trend(timeseries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeseries.rolling(window=size).mean()
    # 对size个数据移动平均的方差
    rol_std = timeseries.rolling(window=size).std()

    timeseries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_std.plot(color='black', label='Rolling standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()


def draw_ts(timeseries):
    f = plt.figure(facecolor='white')
    timeseries.plot(color='blue')
    plt.show()


# Dickey-Fuller test:
def teststationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput


# diff_data = ts.diff()
# diff_data.dropna(inplace=True)

# train = train.diff(1)
# train = train.dropna()

for index in range(0, 2):
    i = companyList[index]
    print(i)
    ts = Arima(i)
    ts = ts.reset_index(drop=True)
    print(len(ts))
    train = ts[0:233]
    train = train.dropna()
    model = ARIMA(train, order=(5, 1, 1))
    model_fit = model.fit(disp=-1)
    print(model_fit.summary())
    prediction = model_fit.forecast(15)[0]
    x = range(233, 248)
    plt.plot(ts)
    plt.plot(x, prediction, color='red')
    plt.show()
    joblib.dump(model_fit, "models/" + i + ".pkl")
    print(index)

# prediction= model_fit.forecast(15)[0]
# x = range(233,248)
#
#
# plt.plot(ts)
# plt.plot(x,prediction, color='red')
# plt.show()

# predictions = []
# for i in range(100,len(ts)):
#     model = ARIMA(ts[0:i], order=(1, 1, 1))
#     model_fit = model.fit(disp=0)
#
#     prediction = model_fit.forecast(5)
#     print(prediction)
#     predictions.append(prediction[0][3])
# print(predictions)
# plt.plot(ts)
# x = range(100,250)
# plt.plot(x,predictions, color='red')
# plt.show()
