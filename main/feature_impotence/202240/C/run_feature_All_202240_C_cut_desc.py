import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)
from config.Config import File202240Config
from common.feature_common.feature_cut_run import *
from config.Enum import feature_C_desc

file_config = File202240Config(p)

"""
依次去掉不重要的TSS特征
"""
feature_importance = feature_C_desc

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['NN'][i])
    NN_run(p, file_config, '202240', 'C', feature_cut_list, 'desc')

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['LSTM'][i])
    LSTM_run(p, file_config, '202240', 'C', feature_cut_list, 'desc')

# feature_cut_list = []
# for i in range(9):
#     feature_cut_list.append(feature_importance['LSTM_Att'][i])
#     LSTM_attention_run(p, file_config, '202240', 'C', feature_cut_list, 'desc')

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['LSTM_Att2'][i])
    LSTM_attention2_run(p, file_config, '202240', 'C', feature_cut_list, 'desc')

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['BLSTM'][i])
    Bi_LSTM_run(p, file_config, '202240', 'C', feature_cut_list, 'desc')

# feature_cut_list = []
# for i in range(9):
#     feature_cut_list.append(feature_importance['BLSTM_Att'][i])
#     Bi_LSTM_attention_run(p, file_config, '202240', 'C', feature_cut_list, 'desc')

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['BLSTM_Att2'][i])
    Bi_LSTM_attention2_run(p, file_config, '202240', 'C', feature_cut_list, 'desc')

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['GRU'][i])
    GRU_run(p, file_config, '202240', 'C', feature_cut_list, 'desc')

# feature_cut_list = []
# for i in range(9):
#     feature_cut_list.append(feature_importance['GRU_Att'][i])
#     GRU_attention_run(p, file_config, '202240', 'C', feature_cut_list, 'desc')

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['GRU_Att2'][i])
    GRU_attention2_run(p, file_config, '202240', 'C', feature_cut_list, 'desc')

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['BGRU'][i])
    Bi_GRU_run(p, file_config, '202240', 'C', feature_cut_list, 'desc')

# feature_cut_list = []
# for i in range(9):
#     feature_cut_list.append(feature_importance['BGRU_Att'][i])
#     Bi_GRU_attention_run(p, file_config, '202240', 'C', feature_cut_list, 'desc')

feature_cut_list = []
for i in range(9):
    feature_cut_list.append(feature_importance['BGRU_Att2'][i])
    Bi_GRU_attention2_run(p, file_config, '202240', 'C', feature_cut_list, 'desc')
