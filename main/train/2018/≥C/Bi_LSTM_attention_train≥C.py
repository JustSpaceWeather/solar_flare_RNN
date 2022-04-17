import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import File2018Config
from common.train_common.C.Bi_LSTM_attention_train_C import Bi_LSTM_attention_C_train

file_config = File2018Config(p)
Bi_LSTM_attention_C_train(p, file_config, '2018')
