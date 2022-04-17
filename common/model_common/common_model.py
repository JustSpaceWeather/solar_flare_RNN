import tensorflow as tf
from keras.initializers import glorot_normal
from keras.layers import Dense, Dropout, BatchNormalization, Activation


def common_NN(dropout_rate, seed):
    return [
        # 第一层
        Dense(
            units=128,
            kernel_initializer=glorot_normal(seed),
            bias_initializer=tf.zeros_initializer()
        ),
        Dropout(dropout_rate, seed=seed),
        BatchNormalization(),
        Activation('relu'),
        # 第二层
        Dense(
            units=64,
            kernel_initializer=glorot_normal(seed),
            bias_initializer=tf.zeros_initializer()
        ),
        Dropout(dropout_rate, seed=seed),
        BatchNormalization(),
        Activation('relu'),
        # 第三层输出层
        Dense(
            units=2,
            activation='softmax',
            kernel_initializer=glorot_normal(seed),
            bias_initializer=tf.zeros_initializer()
        ),
    ]
