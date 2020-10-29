import tensorflow as tf


class StockMovementPrediction(tf.keras.Model):
    def __init__(self, training_days, predict_days, feature_num):
        super(StockMovementPrediction, self).__init__()

        #增加层，增大学习率，增加可用天数
        self.lstm1 = tf.keras.layers.LSTM(100,
                                          activation="tanh",
                                          return_sequences=True,
                                          input_shape=(training_days, feature_num))
        self.lstm2 = tf.keras.layers.LSTM(80, activation='tanh',return_sequences=True)
        self.lstm3 = tf.keras.layers.LSTM(40, activation='tanh')
        self.dense = tf.keras.layers.Dense(predict_days)

    def call(self, x):
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.lstm3(x)

        pre_acti=self.dense(x)

        return pre_acti,tf.keras.activations.tanh(pre_acti)
