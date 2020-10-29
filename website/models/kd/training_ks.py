# -*- coding: utf-8 -*-
# @Time    : 21:52 2020/10/28 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : training_ks.py
import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import utils

from ks_model import KnowledgeDistillationModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

pd.set_option('display.max_columns', None)

config = utils.read_config()

predict_days = config['predict_days']

ticker_file_path = config['ticker_file']
tickers = utils.get_tikcer(ticker_file_path)

feature_num = config['feature_num']

lr = config['lr']
clip_grad = config['clipnorm']
batch_size = config['batch_size']
EPOCHES = config['epoches']
alpha = config['alpha']
beta = config['beta']

# ------------------------------------training model------------------------------------
for name in tickers:
    model = KnowledgeDistillationModel(predict_days)

    loss_object = tf.keras.losses.MeanSquaredError(name='training_loss_fn')

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipvalue=clip_grad)

    split_path = config['ks_training_path']
    train_x_path = '{}/{}/{}'.format(split_path, name, 'training_x.npy')
    train_y_path = '{}/{}/{}'.format(split_path, name, 'training_y.npy')
    train_soft_y_path = '{}/{}/{}'.format(split_path, name, 'training_soft_y.npy')

    X = np.load(train_x_path)
    y = np.load(train_y_path)
    soft_y = np.load(train_soft_y_path)

    # init_data_path=config['init_data_path']
    # np.save(init_data_path,X)

    training_data = tf.data.Dataset.from_tensor_slices((X, y, soft_y)).batch(batch_size)

    training_loss = tf.keras.metrics.Mean(name='total_training_loss')

    for epoch in range(EPOCHES):
        training_loss.reset_states()

        for idx, (training_x, training_y, training_soft_y) in enumerate(training_data):
            with tf.GradientTape() as tape:
                pred = model(training_x)
                loss = alpha * loss_object(y_true=training_y, y_pred=pred) + beta * loss_object(y_true=training_soft_y, y_pred=pred)
            gradients = tape.gradient(target=loss, sources=model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            training_loss.update_state(loss)
            if idx % 20 == 0:
                logging.info('Epoch: {}, training_loss: {:.3f}'.format(epoch, training_loss.result().numpy()))

    model_path = config['ks_model_path'].format(name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model.save_weights(filepath=model_path, save_format='tf')
    logging.info('model saved to {}'.format(model_path))
