import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import File202240Config
from common.train_common.M.LSTM_attention2_train_M import LSTM_attention2_M_train

file_config = File202240Config(p)
LSTM_attention2_M_train(p, file_config, '202240')
