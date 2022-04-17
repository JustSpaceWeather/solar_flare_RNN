import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import TTFileConfig
from common.train_common.C.Bi_GRU_attention2_train_C import Bi_GRU_attention2_C_train

file_config = TTFileConfig(p)

# NN_C_train(p, file_config, 'TT')
# LSTM_C_train(p, file_config, 'TT')
# LSTM_attention_C_train(p, file_config, 'TT')
# LSTM_attention2_C_train(p, file_config, 'TT')
# GRU_C_train(p, file_config, 'TT')
# GRU_attention_C_train(p, file_config, 'TT')
# GRU_attention2_C_train(p, file_config, 'TT')
# Bi_LSTM_C_train(p, file_config, 'TT')
# Bi_LSTM_attention_C_train(p, file_config, 'TT')
# Bi_LSTM_attention2_C_train(p, file_config, 'TT')
# Bi_GRU_C_train(p, file_config, 'TT')
# Bi_GRU_attention_C_train(p, file_config, 'TT')
Bi_GRU_attention2_C_train(p, file_config, 'TT')
