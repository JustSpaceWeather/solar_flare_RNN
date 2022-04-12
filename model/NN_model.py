import keras
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.initializers import glorot_normal


def get_NN_model(learning_rate: float, dropout_rate: float, seed: int, score_metrics: list):
    """
    :param learning_rate: 学习率
    :param dropout_rate: 神经元失活率
    :param seed: seed: 随机数种子  Glorot正态分布初始化方法和Dropout
    :param score_metrics: 评价指标

    使用NN算法，三层全连接层，前两层使用relu激活函数，最后一层为输出层，使用softmax函数
    由于本问题为二分类问题，输出层output_size=2
    学习优化算法为Adam，损失函数为交叉熵

    :return: 编译好的NN模型
    """
    keras.initializers.he_normal(521)
    model = Sequential([
        InputLayer(input_shape=(10,)),
        # 第一层
        Dense(
            units=128,
            kernel_initializer=glorot_normal(seed)
        ),
        Dropout(dropout_rate, seed=seed),
        BatchNormalization(),
        Activation('relu'),
        # 第二层
        Dense(
            units=64,
            kernel_initializer=glorot_normal(seed)
        ),
        Dropout(dropout_rate, seed=seed),
        BatchNormalization(),
        Activation('relu'),
        # 第三层输出层
        Dense(
            units=2,
            activation='softmax',
            kernel_initializer=glorot_normal(seed)
        ),
    ])
    adam = Adam(learning_rate)
    model.compile(
        optimizer=adam,
        loss='binary_crossentropy',  # 'categorical_crossentropy',  # 交叉熵
        metrics=score_metrics  # [Accuracy, TSS, FAR, HSS, Precision, Recall]  # 准确率,
    )
    return model
