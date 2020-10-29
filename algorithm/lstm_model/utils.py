import numpy as np
import json
import pandas as pd


def read_config():
    '''
    read config file content

    :return: config dictionary
    '''
    with open('config.json', 'r', encoding='utf8') as fr:
        config = json.load(fr)
    return config

def read_df(file_path, tsv_header):
    '''
    read dataframe from tsv file and do some adaptions

    :return: dataframe
    '''
    df = pd.read_csv(file_path, delimiter='\t', header=None)
    #drop blank columns
    df.drop([11,12],axis=1,inplace=True)
    # set column headers
    df.columns = tsv_header
    # convert date type
    df['date'] = pd.to_datetime(df['date'])
    # set dataframe index
    df.set_index('date', inplace=True)
    # remove irrelevant columns
    df.drop([ 'COMNAM', 'SHROUT'], axis=1, inplace=True)

    dfs_grouped_company=df.groupby('TICKER')
    return dfs_grouped_company

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


def nomalize_df(df,norm_header):
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


def split_ks_rolling_series(seq):
    '''
    producce rolling data series for LSTM.

    :param seq: dataframe of stock information.
    :param in_num: training days number.
    :param out_num: predict days number.
    :return: training data.
    '''
    X, y = [], []
    for i in range(len(seq)):
        X.append(seq[i])

    return np.vstack(X)

def split_rolling_series(seq, in_num, out_num):
    '''
    producce rolling data series for LSTM.

    :param seq: dataframe of stock information.
    :param in_num: training days number.
    :param out_num: predict days number.
    :return: training data.
    '''
    X, y = [], []
    for i in range(len(seq)):
        end = i + in_num
        o_end = end + out_num

        if o_end > len(seq):
            break

        seq_x, seq_y = seq[i:end, :], seq[end:o_end, 3]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

def get_tikcer(file_path):
    tickers=list()
    with open(file_path,'r',encoding='utf8') as fr:
        for line  in fr:
            tickers.append(line.strip())

    return tickers

def get_ticker_scalar_info(file_path):
    with open(file_path,'r',encoding='utf8') as fr:
        return json.load(fr)