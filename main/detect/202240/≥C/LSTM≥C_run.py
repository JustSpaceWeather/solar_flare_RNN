import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.LSTM_run import LSTM
from config.Config import File202240Config

file_config = File202240Config(p)
LSTM(p, file_config, '202240', 'C')
