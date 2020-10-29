import talib
import pandas as pd
import csv

dataset_ex_df = pd.read_csv('transaction_data.tsv', sep='\t', header=0)
companyList = dataset_ex_df['TICKER'].unique().tolist()
sentiment = pd.read_csv('merged_sentiment.tsv', sep='\t', header=0)
print(companyList)


def calculate(company):
    alldata = dataset_ex_df[dataset_ex_df.TICKER == company]
    data = alldata['PRC']
    print(data)
    MA10 = talib.SMA(data, 10)
    MA20 = talib.SMA(data, 20)
    MA30 = talib.SMA(data, 30)
    DIFF, DEA, MACD = talib.MACD(data, fastperiod=12, slowperiod=26, signalperiod=9)
    RSI6 = talib.RSI(data, 6)
    RSI12 = talib.RSI(data, 12)
    RSI24 = talib.RSI(data, 24)
    MFI = talib.MFI(alldata['BIDLO'], alldata['ASKHI'], alldata['PRC'], alldata['VOL'])
    return MA10, MA20, MA30, DIFF, DEA, MACD, RSI6, RSI12, RSI24, MFI


# with open('indicator.tsv', 'w+', newline='') as csvfile:
#     writer = csv.writer(csvfile)
ma10 = []
ma20 = []
ma30 = []
diff = []
dea = []
macd = []
rsi6 = []
rsi12 = []
rsi24 = []
mfi = []
for i in companyList:
    MA10, MA20, MA30, DIFF, DEA, MACD, RSI6, RSI12, RSI24, MFI = calculate(i)
    MA10 = MA10.tolist()
    MA20 = MA20.tolist()
    MA30 = MA30.tolist()
    DIFF = DIFF.tolist()
    DEA = DEA.tolist()
    MACD = MACD.tolist()
    RSI6 = RSI6.tolist()
    RSI12 = RSI12.tolist()
    RSI24 = RSI24.tolist()
    MFI = MFI.tolist()
    ma10 = ma10 + MA10
    ma20 = ma20 + MA20
    ma30 = ma30 + MA30
    diff = diff + DIFF
    dea = dea + DEA
    macd = macd + MACD
    rsi6 = rsi6 + RSI6
    rsi12 = rsi12 + RSI12
    rsi24 = rsi24 + RSI24
    mfi = mfi + MFI
data = {'MA10': ma10, 'MA20': ma20, 'MA30': ma30,
        'DIFF': diff, 'DEA': dea, 'MACD': macd,
        'RSI6': rsi6, 'RSI12': rsi12, 'RSI24': rsi24,
        'MFI': mfi}
dataframe = pd.DataFrame(data)
print(dataframe)
print(dataset_ex_df)

res = pd.concat([sentiment, dataframe], axis=1)

print(res)

res.to_csv("indicator.tsv", mode='w', sep='\t', header=True, index=False)

