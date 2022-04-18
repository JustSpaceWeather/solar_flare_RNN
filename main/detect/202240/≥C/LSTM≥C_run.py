import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.LSTM_run import LSTM
from config.Config import TTFileConfig

file_config = TTFileConfig(p)
LSTM(p, file_config, '2022', 'C')
