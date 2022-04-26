import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)
from config.Config import File2022Config
from common.train_common.train import *

file_config = File2022Config(p)

NN_train(p, file_config, '2022', 'C')
LSTM_train(p, file_config, '2022', 'C')
LSTM_attention_train(p, file_config, '2022', 'C')
LSTM_attention2_train(p, file_config, '2022', 'C')
GRU_train(p, file_config, '2022', 'C')
GRU_attention_train(p, file_config, '2022', 'C')
GRU_attention2_train(p, file_config, '2022', 'C')
Bi_LSTM_train(p, file_config, '2022', 'C')
Bi_LSTM_attention_train(p, file_config, '2022', 'C')
Bi_LSTM_attention2_train(p, file_config, '2022', 'C')
Bi_GRU_train(p, file_config, '2022', 'C')
Bi_GRU_attention_train(p, file_config, '2022', 'C')
Bi_GRU_attention2_train(p, file_config, '2022', 'C')
