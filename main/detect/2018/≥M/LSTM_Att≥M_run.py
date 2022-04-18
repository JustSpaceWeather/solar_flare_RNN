import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.LSTM_Att_run import LSTM_Att
from config.Config import TTFileConfig

file_config = TTFileConfig(p)
LSTM_Att(p, file_config, '2018', 'M')
