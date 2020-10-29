import numpy as np
import json


def read_config():
    '''
    read config file content

    :return: config dictionary
    '''
    with open('config.json', 'r', encoding='utf8') as fr:
        config = json.load(fr)
    return config


# -------------------------------------prepcrocessing--------------------------------
def get_price_minmax_scalar_param(df):
    '''
    compute min value and max value of dataframe price columns.
    This is the prepration for reverse data nomalization.

    :param df: dataframe of stock information
    :return: min price and max price of price column.
    '''
    prc_columns = np.array(df['PRC'].tolist(), np.float64)
    max_prc, min_prc = np.nanmax(prc_columns), np.nanmin(prc_columns)

    return min_prc, max_prc


def nomalize_df(df, norm_header):
    '''
    nomalization for data.

    :param df: dataframe of stock information.
    :return: nomalized dataframe.
    '''
    # normalization
    for col in norm_header:
        cur_columns = np.array(df[col].tolist(), np.float64)
        min_value, max_value = np.nanmin(cur_columns), np.nanmax(cur_columns)

        df[col] = df[col].apply(lambda x: (x - min_value) / (max_value - min_value) * 2. - 1.)

    return df


def split_rolling_series(seq):
    '''
    producce rolling data series for LSTM.

    :param seq: dataframe of stock information.
    :param in_num: training days number.
    :param out_num: predict days number.
    :return: training data.
    '''
    X, y = [], []
    for i in range(len(seq)):
        if i + i > len(seq):
            break

        seq_x = seq[i, :]
        seq_y = seq[i + 1, 3]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def get_tikcer(file_path):
    tickers = list()
    with open(file_path, 'r', encoding='utf8') as fr:
        for line in fr:
            tickers.append(line.strip())

    return tickers


def get_ticker_scalar_info(file_path):
    with open(file_path, 'r', encoding='utf8') as fr:
        return json.load(fr)
