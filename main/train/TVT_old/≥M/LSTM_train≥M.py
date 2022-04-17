import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import TVTFileConfig
from common.train_common.M.LSTM_train_M import LSTM_M_train

file_config = TVTFileConfig(p)
LSTM_M_train(p, file_config, 'TVT')
