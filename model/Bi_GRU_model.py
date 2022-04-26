import keras
import tensorflow as tf
from keras.initializers import glorot_normal
from keras.layers import InputLayer, Bidirectional, GRU, Dense, Dropout, BatchNormalization, Activation
from keras.models import Sequential
from keras.optimizers import Adam

from common.model_common.common_model import common_NN


def get_Bi_GRU_model(time_steps: int,
                     learning_rate: float,
                     dropout_rate: float,
                     seed: int,
                     score_metrics: list,
                     feature_size=10):
    """
    获得编译好的双向GRU模型
    :param time_steps: 时间步
    :param learning_rate: 学习率
    :param dropout_rate: 神经元失活率
    :param seed: 随机数种子  Glorot正态分布初始化方法和Dropout
    :param score_metrics: 评价指标
    :param feature_size: 特征维度
    :return: 编译好的双向GRU模型
    """
    keras.initializers.he_normal(521)
    model = Sequential(
        [
            InputLayer(input_shape=(time_steps, feature_size)),
            # 第一层Bi-GRU
            Bidirectional(
                GRU(
                    units=256,
                    kernel_initializer=glorot_normal(seed),
                    activation='tanh',
                    bias_initializer=tf.zeros_initializer()
                ),
            ),
            Dropout(dropout_rate, seed=seed),
            BatchNormalization(),
            # 第二层
            Dense(
                units=128,
                kernel_initializer=glorot_normal(seed),
                bias_initializer=tf.zeros_initializer()
            ),
            Dropout(dropout_rate, seed=seed),
            BatchNormalization(),
            Activation('relu'),
        ] + common_NN(dropout_rate, seed)
    )
    adam = Adam(learning_rate)
    model.compile(
        optimizer=adam,
        loss='binary_crossentropy',  # 'categorical_crossentropy',  # 交叉熵
        metrics=score_metrics  # 准确率,
    )
    return model
