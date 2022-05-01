from common.detect_common.detect import detect
from common.detect_common.same.detect_same import detect_same
from config.Enum import ModelType
# 加载模型
from model.Bi_GRU_attention2_model import get_Bi_GRU_attention_model as get_Bi_GRU_attention2_model
from model.Bi_GRU_attention_model import get_Bi_GRU_attention_model
from model.Bi_GRU_model import get_Bi_GRU_model
from model.Bi_LSTM_attention2_model import get_Bi_LSTM_attention_model as get_Bi_LSTM_attention2_model
from model.Bi_LSTM_attention_model import get_Bi_LSTM_attention_model
from model.Bi_LSTM_model import get_Bi_LSTM_model
from model.GRU_attention2_model import get_GRU_attention_model as get_GRU_attention2_model
from model.GRU_attention_model import get_GRU_attention_model
from model.GRU_model import get_GRU_model
from model.LSTM_attention2_model import get_LSTM_attention_model as get_LSTM_attention2_model
from model.LSTM_attention_model import get_LSTM_attention_model
from model.LSTM_model import get_LSTM_model
from model.NN_model import get_NN_model

model_type = ModelType()


def get_model_NN(time_steps, learning_rate, dropout_rate, seed, score_metrics):
    model = get_NN_model(
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        seed=seed,
        score_metrics=score_metrics
    )
    return model


def NN(p, file_config, detect_type, class_type: str, mode='normal') -> None:
    """
    :param p: 项目根目录地址
    :param file_config: 训练文件配置类对象
    :param detect_type: 数据集类型  TT TVT 2018 2022
    :param class_type: 分类类型  C  M
    :param mode: normal:不相同混淆矩阵，same：相同混淆矩阵
    """
    if mode == 'same':
        detect_same(p, file_config, detect_type, class_type, model_type.NN, get_model_NN)
    else:
        detect(p, file_config, detect_type, class_type, model_type.NN, get_model_NN)


def LSTM(p, file_config, detect_type, class_type: str, mode='normal') -> None:
    if mode == 'same':
        detect_same(p, file_config, detect_type, class_type, model_type.LSTM, get_LSTM_model)
    else:
        detect(p, file_config, detect_type, class_type, model_type.LSTM, get_LSTM_model)


def LSTM_Att(p, file_config, detect_type, class_type: str, mode='normal') -> None:
    if mode == 'same':
        detect_same(p, file_config, detect_type, class_type, model_type.LSTM_attention, get_LSTM_attention_model)
    else:
        detect(p, file_config, detect_type, class_type, model_type.LSTM_attention, get_LSTM_attention_model)


def LSTM_Att2(p, file_config, detect_type, class_type: str, mode='normal') -> None:
    if mode == 'same':
        detect_same(p, file_config, detect_type, class_type, model_type.LSTM_attention2, get_LSTM_attention2_model)
    else:
        detect(p, file_config, detect_type, class_type, model_type.LSTM_attention2, get_LSTM_attention2_model)


def BLSTM(p, file_config, detect_type, class_type: str, mode='normal') -> None:
    if mode == 'same':
        detect_same(p, file_config, detect_type, class_type, model_type.Bi_LSTM, get_Bi_LSTM_model)
    else:
        detect(p, file_config, detect_type, class_type, model_type.Bi_LSTM, get_Bi_LSTM_model)


def BLSTM_Att(p, file_config, detect_type, class_type: str, mode='normal') -> None:
    if mode == 'same':
        detect_same(p, file_config, detect_type, class_type, model_type.Bi_LSTM_attention, get_Bi_LSTM_attention_model)
    else:
        detect(p, file_config, detect_type, class_type, model_type.Bi_LSTM_attention, get_Bi_LSTM_attention_model)


def BLSTM_Att2(p, file_config, detect_type, class_type: str, mode='normal') -> None:
    if mode == 'same':
        detect_same(p, file_config, detect_type, class_type, model_type.Bi_LSTM_attention2,
                    get_Bi_LSTM_attention2_model)
    else:
        detect(p, file_config, detect_type, class_type, model_type.Bi_LSTM_attention2, get_Bi_LSTM_attention2_model)


def GRU(p, file_config, detect_type, class_type: str, mode='normal') -> None:
    if mode == 'same':
        detect_same(p, file_config, detect_type, class_type, model_type.GRU, get_GRU_model)
    else:
        detect(p, file_config, detect_type, class_type, model_type.GRU, get_GRU_model)


def GRU_Att(p, file_config, detect_type, class_type: str, mode='normal') -> None:
    if mode == 'same':
        detect_same(p, file_config, detect_type, class_type, model_type.GRU_attention, get_GRU_attention_model)
    else:
        detect(p, file_config, detect_type, class_type, model_type.GRU_attention, get_GRU_attention_model)


def GRU_Att2(p, file_config, detect_type, class_type: str, mode='normal') -> None:
    if mode == 'same':
        detect_same(p, file_config, detect_type, class_type, model_type.GRU_attention2, get_GRU_attention2_model)
    else:
        detect(p, file_config, detect_type, class_type, model_type.GRU_attention2, get_GRU_attention2_model)


def BGRU(p, file_config, detect_type, class_type: str, mode='normal') -> None:
    if mode == 'same':
        detect_same(p, file_config, detect_type, class_type, model_type.Bi_GRU, get_Bi_GRU_model)
    else:
        detect(p, file_config, detect_type, class_type, model_type.Bi_GRU, get_Bi_GRU_model)


def BGRU_Att(p, file_config, detect_type, class_type: str, mode='normal') -> None:
    if mode == 'same':
        detect_same(p, file_config, detect_type, class_type, model_type.Bi_GRU_attention, get_Bi_GRU_attention_model)
    else:
        detect(p, file_config, detect_type, class_type, model_type.Bi_GRU_attention, get_Bi_GRU_attention_model)


def BGRU_Att2(p, file_config, detect_type, class_type: str, mode='normal') -> None:
    if mode == 'same':
        detect_same(p, file_config, detect_type, class_type, model_type.Bi_GRU_attention2, get_Bi_GRU_attention2_model)
    else:
        detect(p, file_config, detect_type, class_type, model_type.Bi_GRU_attention2, get_Bi_GRU_attention2_model)
