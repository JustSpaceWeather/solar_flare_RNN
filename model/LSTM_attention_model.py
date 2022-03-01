from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_normal
from layer.attention import Attention


def get_LSTM_attention_model(time_steps, learning_rate: float, dropout_rate: float, glorot_normal_seed: int,
                             score_metrics: list):
    """
    获得编译好的lstm模型
    :param time_steps: 时间步
    :param learning_rate: 学习率
    :param dropout_rate: 神经元失活率
    :param glorot_normal_seed: Glorot正态分布初始化方法随机数种子
    :param score_metrics: 评价指标
    :return: 编译好的lstm模型
    """
    keras.initializers.he_normal(521)
    model = Sequential([
        InputLayer(input_shape=(time_steps, 10)),
        # 第一层LSTM
        LSTM(
            units=128,
            kernel_initializer=glorot_normal(glorot_normal_seed),
            activation='tanh'
        ),
        Dropout(dropout_rate),
        BatchNormalization(),
        Attention(128),
        # 第二层
        Dense(
            units=128,
            kernel_initializer=glorot_normal(glorot_normal_seed)
        ),
        Dropout(dropout_rate),
        BatchNormalization(),
        Activation('relu'),
        # 第三层
        Dense(
            units=64,
            kernel_initializer=glorot_normal(glorot_normal_seed)
        ),
        Dropout(dropout_rate),
        BatchNormalization(),
        Activation('relu'),
        # 第四层输出层
        Dense(
            units=2,
            activation='softmax',
            kernel_initializer=glorot_normal(glorot_normal_seed)
        ),
    ])
    adam = Adam(learning_rate)
    model.compile(
        optimizer=adam,
        loss='categorical_crossentropy',  # 交叉熵
        metrics=score_metrics
    )
    return model
