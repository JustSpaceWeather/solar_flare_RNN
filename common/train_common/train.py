from common.train_common.train_common import train
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


def get_model_NN(time_steps, learning_rate, dropout_rate, seed, score_metrics):
    return get_NN_model(learning_rate, dropout_rate, seed, score_metrics)


def NN_train(p: str, file_config, data_type: str, class_type):
    train(p, file_config, data_type, model_type.NN, class_type, get_model_NN)


# LSTM系列
def LSTM_train(p: str, file_config, data_type: str, class_type):
    train(p, file_config, data_type, model_type.LSTM, class_type, get_LSTM_model)


def LSTM_attention_train(p: str, file_config, data_type: str, class_type):
    train(p, file_config, data_type, model_type.LSTM_attention, class_type, get_LSTM_attention_model)


def LSTM_attention2_train(p: str, file_config, data_type: str, class_type):
    train(p, file_config, data_type, model_type.LSTM_attention2, class_type, get_LSTM_attention2_model)


# Bi-LSTM系列
def Bi_LSTM_train(p: str, file_config, data_type: str, class_type):
    train(p, file_config, data_type, model_type.Bi_LSTM, class_type, get_Bi_LSTM_model)


def Bi_LSTM_attention_train(p: str, file_config, data_type: str, class_type):
    train(p, file_config, data_type, model_type.Bi_LSTM_attention, class_type, get_Bi_LSTM_attention_model)


def Bi_LSTM_attention2_train(p: str, file_config, data_type: str, class_type):
    train(p, file_config, data_type, model_type.Bi_LSTM_attention2, class_type, get_Bi_LSTM_attention2_model)


# GRU系列
def GRU_train(p: str, file_config, data_type: str, class_type):
    train(p, file_config, data_type, model_type.GRU, class_type, get_GRU_model)


def GRU_attention_train(p: str, file_config, data_type: str, class_type):
    train(p, file_config, data_type, model_type.GRU_attention, class_type, get_GRU_attention_model)


def GRU_attention2_train(p: str, file_config, data_type: str, class_type):
    train(p, file_config, data_type, model_type.GRU_attention2, class_type, get_GRU_attention2_model)


# Bi-GRU系列
def Bi_GRU_train(p: str, file_config, data_type: str, class_type):
    train(p, file_config, data_type, model_type.Bi_GRU, class_type, get_Bi_GRU_model)


def Bi_GRU_attention_train(p: str, file_config, data_type: str, class_type):
    train(p, file_config, data_type, model_type.Bi_GRU_attention, class_type, get_Bi_GRU_attention_model)


def Bi_GRU_attention2_train(p: str, file_config, data_type: str, class_type):
    train(p, file_config, data_type, model_type.Bi_GRU_attention2, class_type, get_Bi_GRU_attention2_model)
