import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)
from config.Config import File202240Config
from common.feature_common.feature_train import *

file_config = File202240Config(p)

# NN_train(p, file_config, '202240', 'C')
# LSTM_train(p, file_config, '202240', 'C')
# LSTM_attention_train(p, file_config, '202240', 'C')
LSTM_attention2_train(p, file_config, '202240', 'C')
# GRU_train(p, file_config, '202240', 'C')
# GRU_attention_train(p, file_config, '202240', 'C')
GRU_attention2_train(p, file_config, '202240', 'C')
# Bi_LSTM_train(p, file_config, '202240', 'C')
# Bi_LSTM_attention_train(p, file_config, '202240', 'C')
Bi_LSTM_attention2_train(p, file_config, '202240', 'C')
# Bi_GRU_train(p, file_config, '202240', 'C')
# Bi_GRU_attention_train(p, file_config, '202240', 'C')
Bi_GRU_attention2_train(p, file_config, '202240', 'C')


# NN_train(p, file_config, '202240', 'M')
# LSTM_train(p, file_config, '202240', 'M')
# LSTM_attention_train(p, file_config, '202240', 'M')
LSTM_attention2_train(p, file_config, '202240', 'M')
# GRU_train(p, file_config, '202240', 'M')
# GRU_attention_train(p, file_config, '202240', 'M')
GRU_attention2_train(p, file_config, '202240', 'M')
# Bi_LSTM_train(p, file_config, '202240', 'M')
# Bi_LSTM_attention_train(p, file_config, '202240', 'M')
Bi_LSTM_attention2_train(p, file_config, '202240', 'M')
# Bi_GRU_train(p, file_config, '202240', 'M')
# Bi_GRU_attention_train(p, file_config, '202240', 'M')
Bi_GRU_attention2_train(p, file_config, '202240', 'M')
