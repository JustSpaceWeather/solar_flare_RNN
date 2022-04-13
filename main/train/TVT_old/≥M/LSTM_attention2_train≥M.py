import sys
import os

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import TVTFileConfig
from main.train_common.M.LSTM_attention2_train_M import LSTM_attention2_M_train

file_config = TVTFileConfig(p)
LSTM_attention2_M_train(p, file_config, 'TVT')
