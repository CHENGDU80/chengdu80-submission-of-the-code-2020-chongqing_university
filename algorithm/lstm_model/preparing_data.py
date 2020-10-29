# -*- coding: utf-8 -*-
# @Time    : 13:52 2020/10/29 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : preparing_data.py
import json
import logging
import os

import numpy as np
import utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

config = utils.read_config()

file_path=config['file_path']

tsv_header = config['tsv_header']
norm_header = config['norm_header']

training_days = config['training_days']
predict_days = config['predict_days']

train_test_rate = config['train_test_rate']

#--------------------------------------------------------------------------
def dump_scaling_factor(dfs_grouped_company):
    ticker_scala_info = dict()
    for name, company_df in dfs_grouped_company:
        min_prc, max_prc = utils.get_price_minmax_scalar_param(company_df)
        ticker_scala_info[name] = {'min_prc': min_prc, 'max_prc': max_prc}

    with open('./scalar.json', 'w', encoding='utf8') as fw:
        json.dump(ticker_scala_info, fw)

#--------------------------------------------------------------------------
def dump_ticker_names(dfs_grouped_company):
    ticker_names = list()
    for name, company_df in dfs_grouped_company:
        ticker_names.append(name)

    ticker_file_path = config['ticker_file']
    with open(ticker_file_path, 'w', encoding='utf8') as fw:
        for line in ticker_names:
            fw.write(line + '\n')

#--------------------------------------------------------------------------
def split_train_test_file(dfs_grouped_company):
    for name,company_df in dfs_grouped_company:
        logging.info(name)

        company_df = company_df.copy()
        # drop ticker column
        company_df.drop(['TICKER'], axis=1, inplace=True)
        # processing nan value
        company_df.fillna(axis=0, method='backfill', inplace=True)  # fill nan value with the first valid value
        company_df.fillna(0, inplace=True)  # otherwise fill nan value with 0
        # normalizing
        df = utils.nomalize_df(company_df, norm_header)

        X, y = utils.split_rolling_series(df.to_numpy(), training_days, predict_days)  # (147, 100, 7) (147, 5)

        path_prefix = config['split_data_path']
        company_path = (path_prefix + '{}').format(name)
        if not os.path.exists(company_path):
            os.mkdir(company_path)

        train_case_num = len(X) * train_test_rate
        np.save(os.path.join(company_path, 'train_x.npy'), X[:int(train_case_num), :])
        np.save(os.path.join(company_path, 'train_y.npy'), y[:int(train_case_num)])

        np.save(os.path.join(company_path, 'test_x.npy'), X[int(train_case_num):, :])
        np.save(os.path.join(company_path, 'test_y.npy'), y[int(train_case_num):])

        # np.save('./data/init.npy',X) #save as init model data

#--------------------------------------------------------------------------
def dump_ks_test_data(dfs_grouped_company):
    for name, company_df in dfs_grouped_company:
        logging.info(name)

        company_df = company_df.copy()
        company_df.drop(['TICKER'], axis=1, inplace=True)
        df = utils.nomalize_df(company_df, norm_header)
        df.fillna(axis=0, method='backfill', inplace=True)
        df.fillna(0, inplace=True)  # otherwise fill nan value with 0

        df_numpy = df.to_numpy()
        test_num = int(len(df_numpy) * (1 - train_test_rate))
        X = utils.split_ks_rolling_series(df_numpy[-test_num:, :])

        path_prefix = config['split_ks_data_path']
        company_path = (path_prefix + '{}').format(name)
        if not os.path.exists(company_path):
            os.mkdir(company_path)

        np.save(company_path + '/test_x.npy', X)

def main():
    # run once!
    dfs_grouped_company=utils.read_df(file_path,tsv_header)

    split_train_test_file(dfs_grouped_company)

    dump_ks_test_data(dfs_grouped_company)

    dump_scaling_factor(dfs_grouped_company)

    dump_ticker_names(dfs_grouped_company)

if __name__ == '__main__':
    main()
