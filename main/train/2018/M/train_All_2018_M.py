import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)
from config.Config import File2018Config
from common.train_common.train import *

file_config = File2018Config(p)

NN_train(p, file_config, '2018', 'M')
LSTM_train(p, file_config, '2018', 'M')
LSTM_attention_train(p, file_config, '2018', 'M')
LSTM_attention2_train(p, file_config, '2018', 'M')
GRU_train(p, file_config, '2018', 'M')
GRU_attention_train(p, file_config, '2018', 'M')
GRU_attention2_train(p, file_config, '2018', 'M')
Bi_LSTM_train(p, file_config, '2018', 'M')
Bi_LSTM_attention_train(p, file_config, '2018', 'M')
Bi_LSTM_attention2_train(p, file_config, '2018', 'M')
Bi_GRU_train(p, file_config, '2018', 'M')
Bi_GRU_attention_train(p, file_config, '2018', 'M')
Bi_GRU_attention2_train(p, file_config, '2018', 'M')
