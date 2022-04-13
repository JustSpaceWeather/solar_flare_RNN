import sys
import os

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(p)

from config.Config import TTFileConfig
from main.train_common.C.Bi_LSTM_attention2_train_C import Bi_LSTM_attention2_C_train

file_config = TTFileConfig(p)
Bi_LSTM_attention2_C_train(p, file_config, 'TT')
