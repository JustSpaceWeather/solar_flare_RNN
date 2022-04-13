import sys
import os

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import TVTFileConfig
from main.train_common.M.Bi_LSTM_attention_train_M import Bi_LSTM_attention_M_train

file_config = TVTFileConfig(p)
Bi_LSTM_attention_M_train(p, file_config, 'TVT')
