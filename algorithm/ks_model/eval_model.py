import numpy as np
import tensorflow as tf
import utils

from ks_model import KnowledgeDistillationModel

init_X = np.load('./data/init_ks.npy')

config = utils.read_config()

training_days = config['training_days']
predict_days = config['predict_days']
batch_size = config['batch_size']

model_path_prefix = config['ks_model_path']

feature_num = config['feature_num']

init_ds = tf.data.Dataset.from_tensor_slices((init_X)).batch(batch_size)

ticker_scala_path = config['ticker_scalar_path']
ticker_scala_infos = utils.get_ticker_scalar_info(ticker_scala_path)


def get_cur_ticker_minmax_prc(ticker_name):
    max_prc = ticker_scala_infos[ticker_name]['max_prc']
    min_prc = ticker_scala_infos[ticker_name]['min_prc']

    return min_prc, max_prc


def restore_scale_data(converted_data, min_prc, max_prc):
    return (converted_data + 1.) / 2. * (max_prc - min_prc) + min_prc


def get_test_data(ticker_name):
    test_path_prefix = config['ks_testing_path']
    test_path = test_path_prefix + '/' + ticker_name + '/test_x.npy'
    test_x = np.load(test_path)
    return test_x


def get_ks__prediction(company_name):
    # ------------load model---------------
    model_path = model_path_prefix.format(company_name)

    model = KnowledgeDistillationModel(predict_days)

    for x in init_ds:
        _ = model(x, training=False)

    model.load_weights(model_path)

    min_prc, max_prc = get_cur_ticker_minmax_prc(company_name)

    # ----------------model prediction----------

    test_x = get_test_data(company_name)
    test_ds = tf.data.Dataset.from_tensor_slices((test_x)).batch(batch_size)

    all_pred = list()
    for x in test_ds:
        pred = model(x, training=False)
        all_pred.extend(pred.numpy().tolist())

    all_pred = np.array(all_pred, np.float32)

    all_pred = np.squeeze(all_pred)

    all_pred = restore_scale_data(all_pred, min_prc, max_prc)

    kernel=np.squeeze(model.dense.kernel.numpy())
    bias=model.dense.bias.numpy()

    return all_pred.tolist(), kernel,bias


if __name__ == '__main__':
    pred, kernel, bias = get_ks__prediction('A')
    print('pred')
    print(pred)
    print('kernel')
    print(kernel)
    print('bias')
    print(bias)