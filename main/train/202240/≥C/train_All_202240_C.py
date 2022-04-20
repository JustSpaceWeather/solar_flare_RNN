import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)
from config.Config import File202240Config
from common.train_common.C.NN_train_C import NN_C_train
from common.train_common.C.LSTM_train_C import LSTM_C_train
from common.train_common.C.LSTM_attention_train_C import LSTM_attention_C_train
from common.train_common.C.GRU_train_C import GRU_C_train
from common.train_common.C.GRU_attention_train_C import GRU_attention_C_train
from common.train_common.C.Bi_LSTM_train_C import Bi_LSTM_C_train
from common.train_common.C.Bi_LSTM_attention_train_C import Bi_LSTM_attention_C_train
from common.train_common.C.Bi_GRU_train_C import Bi_GRU_C_train
from common.train_common.C.Bi_GRU_attention_train_C import Bi_GRU_attention_C_train

file_config = File202240Config(p)

# NN_C_train(p, file_config, '202240')
LSTM_C_train(p, file_config, '202240')
LSTM_attention_C_train(p, file_config, '202240')
# LSTM_attention2_C_train(p, file_config, '202240')
GRU_C_train(p, file_config, '202240')
GRU_attention_C_train(p, file_config, '202240')
# GRU_attention2_C_train(p, file_config, '202240')
Bi_LSTM_C_train(p, file_config, '202240')
Bi_LSTM_attention_C_train(p, file_config, '202240')
# Bi_LSTM_attention2_C_train(p, file_config, '202240')
Bi_GRU_C_train(p, file_config, '202240')
Bi_GRU_attention_C_train(p, file_config, '202240')
# Bi_GRU_attention2_C_train(p, file_config, '202240')
