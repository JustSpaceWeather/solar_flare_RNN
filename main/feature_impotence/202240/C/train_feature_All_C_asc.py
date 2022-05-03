import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)
from config.Config import File202240Config

file_config = File202240Config(p)
from common.feature_common.feature_cut_train import *

"""
依次去掉重要的TSS特征，从而得到结果
"""
feature_importance = {
    'NN': ['R_VALUE', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTPOT', 'SAVNCPP', 'AREA_ACR', 'ABSNJZH', 'SHRGT45', 'MEANPOT'],
    'LSTM': ['R_VALUE', 'USFLUX', 'TOTUSJH', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
             'SHRGT45'],
    'GRU': ['R_VALUE', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
            'SHRGT45'],
    'BLSTM': ['R_VALUE', 'USFLUX', 'TOTUSJH', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
              'SHRGT45'],
    'BGRU': ['R_VALUE', 'USFLUX', 'TOTUSJH', 'TOTUSJZ', 'SAVNCPP', 'ABSNJZH', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
             'SHRGT45'],
    'LSTM_Att': ['R_VALUE', 'USFLUX', 'TOTUSJH', 'TOTUSJZ', 'SAVNCPP', 'ABSNJZH', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                 'SHRGT45'],
    'GRU_Att': ['R_VALUE', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                'SHRGT45'],
    'BLSTM_Att': ['R_VALUE', 'USFLUX', 'TOTUSJH', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                  'SHRGT45'],
    'BGRU_Att': ['R_VALUE', 'TOTUSJH', 'USFLUX', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                 'SHRGT45'],
    'LSTM_Att2': [],
    'GRU_Att2': [],
    'BLSTM_Att2': [],
    'BGRU_Att2': []
}

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['NN'][i])
    NN_train(p, file_config, '202240', 'C', feature_cut_list)

for i in range(9):
    feature_cut_list.append(feature_importance['LSTM'][i])
    LSTM_train(p, file_config, '202240', 'C', feature_cut_list)

for i in range(9):
    feature_cut_list.append(feature_importance['LSTM_Att'][i])
    LSTM_attention_train(p, file_config, '202240', 'C', feature_cut_list)

for i in range(9):
    feature_cut_list.append(feature_importance['LSTM_Att2'][i])
    LSTM_attention2_train(p, file_config, '202240', 'C', feature_cut_list)

for i in range(9):
    feature_cut_list.append(feature_importance['BLSTM'][i])
    Bi_LSTM_train(p, file_config, '202240', 'C', feature_cut_list)

for i in range(9):
    feature_cut_list.append(feature_importance['BLSTM_Att'][i])
    Bi_LSTM_attention_train(p, file_config, '202240', 'C', feature_cut_list)

for i in range(9):
    feature_cut_list.append(feature_importance['BLSTM_Att2'][i])
    Bi_LSTM_attention2_train(p, file_config, '202240', 'C', feature_cut_list)

for i in range(9):
    feature_cut_list.append(feature_importance['GRU'][i])
    GRU_train(p, file_config, '202240', 'C', feature_cut_list)

for i in range(9):
    feature_cut_list.append(feature_importance['GRU_Att'][i])
    GRU_attention_train(p, file_config, '202240', 'C', feature_cut_list)

for i in range(9):
    feature_cut_list.append(feature_importance['GRU_Att2'][i])
    GRU_attention2_train(p, file_config, '202240', 'C', feature_cut_list)

for i in range(9):
    feature_cut_list.append(feature_importance['BGRU'][i])
    Bi_GRU_train(p, file_config, '202240', 'C', feature_cut_list)

for i in range(9):
    feature_cut_list.append(feature_importance['BGRU_Att'][i])
    Bi_GRU_attention_train(p, file_config, '202240', 'C', feature_cut_list)

for i in range(9):
    feature_cut_list.append(feature_importance['BGRU_Att2'][i])
    Bi_GRU_attention2_train(p, file_config, '202240', 'C', feature_cut_list)
