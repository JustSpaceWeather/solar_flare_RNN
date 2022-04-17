import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import File2018Config
from common.train_common.M.Bi_LSTM_train_M import Bi_LSTM_M_train

file_config = File2018Config(p)
Bi_LSTM_M_train(p, file_config, '2018')
