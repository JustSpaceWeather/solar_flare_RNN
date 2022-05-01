from common.feature_common.feature_run_common import feature_run
from config.Enum import FeatureEnum
from config.Enum import ModelType
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
feature = FeatureEnum()
feature_list = feature.all_feature_list


def get_model_NN(time_steps, learning_rate, dropout_rate, seed, score_metrics, feature_size):
    return get_NN_model(learning_rate, dropout_rate, seed, score_metrics, feature_size)


def NN_feature_run(p: str, file_config, data_type: str, class_type):
    for feature_name in feature_list:
        feature_run(p, file_config, data_type, class_type, model_type.NN, get_model_NN, feature_name)


# LSTM系列
def LSTM_feature_run(p: str, file_config, data_type: str, class_type):
    for feature_name in feature_list:
        feature_run(p, file_config, data_type, class_type, model_type.LSTM, get_LSTM_model, feature_name)


def LSTM_attention_feature_run(p: str, file_config, data_type: str, class_type):
    for feature_name in feature_list:
        feature_run(p, file_config, data_type, class_type, model_type.LSTM_attention, get_LSTM_attention_model,
                    feature_name)


def LSTM_attention2_feature_run(p: str, file_config, data_type: str, class_type):
    for feature_name in feature_list:
        feature_run(p, file_config, data_type, class_type, model_type.LSTM_attention2, get_LSTM_attention2_model,
                    feature_name)


# Bi-LSTM系列
def Bi_LSTM_feature_run(p: str, file_config, data_type: str, class_type):
    for feature_name in feature_list:
        feature_run(p, file_config, data_type, class_type, model_type.Bi_LSTM, get_Bi_LSTM_model, feature_name)


def Bi_LSTM_attention_feature_run(p: str, file_config, data_type: str, class_type):
    for feature_name in feature_list:
        feature_run(p, file_config, data_type, class_type, model_type.Bi_LSTM_attention, get_Bi_LSTM_attention_model,
                    feature_name)


def Bi_LSTM_attention2_feature_run(p: str, file_config, data_type: str, class_type):
    for feature_name in feature_list:
        feature_run(p, file_config, data_type, class_type, model_type.Bi_LSTM_attention2, get_Bi_LSTM_attention2_model,
                    feature_name)


# GRU系列
def GRU_feature_run(p: str, file_config, data_type: str, class_type):
    for feature_name in feature_list:
        feature_run(p, file_config, data_type, class_type, model_type.GRU, get_GRU_model, feature_name)


def GRU_attention_feature_run(p: str, file_config, data_type: str, class_type):
    for feature_name in feature_list:
        feature_run(p, file_config, data_type, class_type, model_type.GRU_attention, get_GRU_attention_model,
                    feature_name)


def GRU_attention2_feature_run(p: str, file_config, data_type: str, class_type):
    for feature_name in feature_list:
        feature_run(p, file_config, data_type, class_type, model_type.GRU_attention2, get_GRU_attention2_model,
                    feature_name)


# Bi-GRU系列
def Bi_GRU_feature_run(p: str, file_config, data_type: str, class_type):
    for feature_name in feature_list:
        feature_run(p, file_config, data_type, class_type, model_type.Bi_GRU, get_Bi_GRU_model, feature_name)


def Bi_GRU_attention_feature_run(p: str, file_config, data_type: str, class_type):
    for feature_name in feature_list:
        feature_run(p, file_config, data_type, class_type, model_type.Bi_GRU_attention, get_Bi_GRU_attention_model,
                    feature_name)


def Bi_GRU_attention2_feature_run(p: str, file_config, data_type: str, class_type):
    for feature_name in feature_list:
        feature_run(p, file_config, data_type, class_type, model_type.Bi_GRU_attention2, get_Bi_GRU_attention2_model,
                    feature_name)
