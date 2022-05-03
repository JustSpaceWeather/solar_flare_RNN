import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)
from config.Config import File202240Config

file_config = File202240Config(p)
from common.feature_common.feature_cut_train import *

"""
依次去掉不重要的TSS特征
"""
feature_importance = {
    'NN': ['MEANPOT', 'SHRGT45', 'ABSNJZH', 'AREA_ACR', 'SAVNCPP', 'TOTPOT', 'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'R_VALUE'],
    'LSTM': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'AREA_ACR', 'SAVNCPP', 'ABSNJZH', 'TOTUSJZ', 'TOTUSJH', 'USFLUX',
             'R_VALUE'],
    'GRU': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'AREA_ACR', 'SAVNCPP', 'ABSNJZH', 'USFLUX', 'TOTUSJZ', 'TOTUSJH',
            'R_VALUE'],
    'BLSTM': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'AREA_ACR', 'SAVNCPP', 'ABSNJZH', 'TOTUSJZ', 'TOTUSJH', 'USFLUX',
              'R_VALUE'],
    'BGRU': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'AREA_ACR', 'ABSNJZH', 'SAVNCPP', 'TOTUSJZ', 'TOTUSJH', 'USFLUX',
             'R_VALUE'],
    'LSTM_Att': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'ABSNJZH', 'SAVNCPP', 'TOTUSJZ', 'TOTUSJH', 'USFLUX',
                 'R_VALUE'],
    'GRU_Att': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'SAVNCPP', 'ABSNJZH', 'USFLUX', 'TOTUSJZ', 'TOTUSJH',
                'R_VALUE'],
    'BLSTM_Att': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'SAVNCPP', 'ABSNJZH', 'TOTUSJZ', 'TOTUSJH', 'USFLUX',
                  'R_VALUE'],
    'BGRU_Att': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'SAVNCPP', 'ABSNJZH', 'TOTUSJZ', 'USFLUX', 'TOTUSJH',
                 'R_VALUE'],
    'LSTM_Att2': [],
    'GRU_Att2': [],
    'BLSTM_Att2': [],
    'BGRU_Att2': []
}

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['NN'][i])
    NN_train(p, file_config, '202240', 'C', feature_cut_list)

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['LSTM'][i])
    LSTM_train(p, file_config, '202240', 'C', feature_cut_list)

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['LSTM_Att'][i])
    LSTM_attention_train(p, file_config, '202240', 'C', feature_cut_list)

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['LSTM_Att2'][i])
    LSTM_attention2_train(p, file_config, '202240', 'C', feature_cut_list)

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['BLSTM'][i])
    Bi_LSTM_train(p, file_config, '202240', 'C', feature_cut_list)

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['BLSTM_Att'][i])
    Bi_LSTM_attention_train(p, file_config, '202240', 'C', feature_cut_list)

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['BLSTM_Att2'][i])
    Bi_LSTM_attention2_train(p, file_config, '202240', 'C', feature_cut_list)

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['GRU'][i])
    GRU_train(p, file_config, '202240', 'C', feature_cut_list)

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['GRU_Att'][i])
    GRU_attention_train(p, file_config, '202240', 'C', feature_cut_list)

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['GRU_Att2'][i])
    GRU_attention2_train(p, file_config, '202240', 'C', feature_cut_list)

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['BGRU'][i])
    Bi_GRU_train(p, file_config, '202240', 'C', feature_cut_list)

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['BGRU_Att'][i])
    Bi_GRU_attention_train(p, file_config, '202240', 'C', feature_cut_list)

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['BGRU_Att2'][i])
    Bi_GRU_attention2_train(p, file_config, '202240', 'C', feature_cut_list)
