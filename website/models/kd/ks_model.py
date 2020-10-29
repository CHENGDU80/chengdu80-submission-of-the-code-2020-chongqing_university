# -*- coding: utf-8 -*-
# @Time    : 21:47 2020/10/28 
# @Author  : Haohao Song
# @Email   : songhaohao2018@cqu.edu.cn
# @File    : ks_model.py

import tensorflow as tf


class KnowledgeDistillationModel(tf.keras.Model):
    def __init__(self, predict_days):
        super(KnowledgeDistillationModel, self).__init__()

        self.dense=tf.keras.layers.Dense(units=predict_days,use_bias=True,activation='tanh')

    def call(self, x):
        pre_acti = self.dense(x)

        return pre_acti
