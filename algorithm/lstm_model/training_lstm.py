import logging
import os

import pandas as pd
import json
import tensorflow as tf
import numpy as np

from lstm_model import StockMovementPrediction
import utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

pd.set_option('display.max_columns', None)

config = utils.read_config()

ticker_file_path=config['ticker_file']
tickers= utils.get_tikcer(ticker_file_path)

train_test_rate=config['train_test_rate']

training_days = config['training_days']
predict_days = config['predict_days']

feature_num = config['feature_num']

lr = config['lr']
clip_grad = config['clipnorm']

batch_size = config['batch_size']

EPOCHES = config['epoches']

#----------------------------training model----------------------------------
for name in tickers:
    model = StockMovementPrediction(training_days, predict_days, feature_num)

    loss_object = tf.keras.losses.MeanSquaredError(name='training_loss_fn')

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=clip_grad)

    split_path = config['split_data_path']
    train_x_path = '{}{}/{}'.format(split_path, name, 'train_x.npy')
    train_y_path = '{}{}/{}'.format(split_path, name, 'train_y.npy')

    X = np.load(train_x_path)
    y = np.load(train_y_path)

    training_data = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)

    training_loss = tf.keras.metrics.Mean(name='total_training_loss')

    for epoch in range(EPOCHES):
        training_loss.reset_states()

        for idx, (training_x, training_y) in enumerate(training_data):
            with tf.GradientTape() as tape:
                _, pred = model(training_x)
                loss = loss_object(y_true=training_y, y_pred=pred)
            gradients = tape.gradient(target=loss, sources=model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            training_loss.update_state(loss)
            if idx % 20 == 0:
                logging.info('Epoch: {}, training_loss: {:.3f}'.format(epoch, training_loss.result().numpy()))

    model_path = config['model_path'].format(name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model.save_weights(filepath=model_path, save_format='tf')
    logging.info('model saved to {}'.format(model_path))

    # --------------------generate ks training data------------------
    #ks training data
    ks_train_x, ks_train_soft_y, ks_train_y = list(), list(), list()
    for training_x, training_y in training_data:
        soft_pred, pred = model(training_x, training=False)

        training_x, training_y = training_x.numpy(), training_y.numpy()
        soft_pred = soft_pred.numpy()

        ks_train_x.append(training_x[:, -1, :])
        ks_train_soft_y.append(soft_pred)
        ks_train_y.append(pred)

    ks_training_path = config['ks_training_path']
    ks_training_path = ks_training_path.format(name)
    if not os.path.exists(ks_training_path):
        os.mkdir(ks_training_path)

    ks_training_x_path = ks_training_path + '/training_x.npy'
    ks_training_soft_y_path = ks_training_path + '/training_soft_y.npy'
    ks_training_y_path = ks_training_path + '/training_y.npy'

    np.save(ks_training_x_path, np.vstack(ks_train_x))  # (133, 17)
    np.save(ks_training_soft_y_path, np.vstack(ks_train_soft_y))  # (133, 1)
    np.save(ks_training_y_path, np.vstack(ks_train_y))  # (133, 1)

'''
# -----------------------------simple evaluation----------------------------------
y_pre = model(np.array(df.tail(training_days)).reshape(1, training_days, feature_num)).numpy()
y_pre = (y_pre + 1.) / 2. * (max_prc - min_prc) + min_prc
y_pre = y_pre[0]

preds = pd.DataFrame(y_pre,
                     index=pd.date_range(start=df.index[-1] + timedelta(days=1), periods=len(y_pre), freq="B"),
                     # business day frequency
                     columns=['PRC'])

print('pred stock price is')
print(preds)
print()

actual_data = pd.read_csv('data/merged_sentiment.tsv', delimiter='\t', header=None, skiprows=200, nrows=predict_days)
headers = ['date', 'TICKER', 'COMNAM', 'BIDLO', 'ASKHI', 'OPENPRC', 'PRC', 'VOL', 'SHROUT', 'POLARITY', 'SUBJECTIVITY']
actual_data.columns = headers
actual_data.set_index('date', inplace=True)  # 索引

actual_data.drop(['TICKER', 'COMNAM', 'BIDLO', 'ASKHI', 'OPENPRC', 'VOL', 'SHROUT', 'POLARITY', 'SUBJECTIVITY'], axis=1,
                 inplace=True)
print('actual stock price is')
print(actual_data)
'''
