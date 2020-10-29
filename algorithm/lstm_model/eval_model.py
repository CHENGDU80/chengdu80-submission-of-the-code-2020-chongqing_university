import os

import numpy as np
import pandas as pd
import tensorflow as tf
import utils

from lstm_model import StockMovementPrediction

init_X = np.load('./data/init.npy')

config = utils.read_config()

training_days = config['training_days']
predict_days = config['predict_days']
batch_size = config['batch_size']

model_path_prefix = config['model_path']

feature_num = config['feature_num']

all_tsv_path = config['file_path']

init_ds = tf.data.Dataset.from_tensor_slices((init_X)).batch(batch_size)

ticker_scala_path = config['ticker_scalar_path']
ticker_scala_infos = utils.get_ticker_scalar_info(ticker_scala_path)


def get_cur_ticker_minmax_prc(ticker_name):
    max_prc = ticker_scala_infos[ticker_name]['max_prc']
    min_prc = ticker_scala_infos[ticker_name]['min_prc']

    return min_prc, max_prc


def restore_scale_data(converted_data, min_prc, max_prc):
    return (converted_data + 1.) / 2. * (max_prc - min_prc) + min_prc


def get_origin_data(ticker_name):
    df = pd.read_csv(all_tsv_path, delimiter='\t', header=None)
    df.drop([11, 12], axis=1, inplace=True)

    tsv_header = config['tsv_header']
    df.columns = tsv_header
    # convert date type
    df['date'] = pd.to_datetime(df['date'])

    dfs_grouped_company = df.groupby('TICKER')

    for name, cur_df in dfs_grouped_company:
        if name == ticker_name:
            return cur_df


def get_lstm__prediction(company_name):
    # ------------load model---------------
    model_path = model_path_prefix.format(company_name)

    model = StockMovementPrediction(training_days, predict_days, feature_num)

    for x in init_ds:
        _, _ = model(x, training=False)

    model.load_weights(model_path, )

    # ----------------model prediction----------
    split_data_prefix = config['split_data_path']
    test_company_path = os.path.join(split_data_prefix, company_name)

    test_x_file_path = os.path.join(test_company_path, 'test_x.npy')
    test_y_file_path = os.path.join(test_company_path, 'test_y.npy')

    test_x = np.load(test_x_file_path)
    test_y = np.load(test_y_file_path)

    test_ds = tf.data.Dataset.from_tensor_slices((test_x)).batch(batch_size)

    all_pred = list()
    for x in test_ds:
        pre_acti, pred = model(x, training=False)
        all_pred.extend(pred.numpy().tolist())

    test_y = np.array(test_y, np.float32)
    all_pred = np.array(all_pred, np.float32)

    test_y = np.squeeze(test_y)
    all_pred = np.squeeze(all_pred)

    # mse_loss=np.mean(np.square(test_y-all_pred))
    # print(len(test_y))#57 0.37664735
    # print(mse_loss)
    # assert 1==2

    min_prc, max_prc = get_cur_ticker_minmax_prc(company_name)
    test_y = restore_scale_data(test_y, min_prc, max_prc)
    all_pred = restore_scale_data(all_pred, min_prc, max_prc)

    cur_df = get_origin_data(company_name)

    train_rate = config['train_test_rate']
    train_case_num = int(len(cur_df) * train_rate)
    train_data = cur_df[config['js_columns']].head(train_case_num)

    test_date = cur_df[['date']].tail(len(cur_df) - train_case_num)

    return train_data, test_date, test_y.tolist(), all_pred.tolist()


if __name__ == '__main__':
    train_data, test_date, ground_true, pred = get_lstm__prediction('A')
    print(train_data)
    print(test_date)
    print('ground_truth')
    print(ground_true)
    print()
    print('pred')
    print(pred)
