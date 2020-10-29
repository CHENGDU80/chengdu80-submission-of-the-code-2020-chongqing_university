import joblib


def getArimaData(company_name):
    file_path = 'models/arima/{}.pkl'.format(company_name)
    loaded = joblib.load(file_path)
    data = loaded.forecast(15)[0]
    return data
