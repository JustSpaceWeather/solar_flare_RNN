import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)
from config.Config import File202240Config
from common.feature_common.feature_run import *

file_config = File202240Config(p)


LSTM_attention2_feature_run(p, file_config, '202240', 'M')
Bi_LSTM_attention2_feature_run(p, file_config, '202240', 'M')
GRU_attention2_feature_run(p, file_config, '202240', 'M')
Bi_GRU_attention2_feature_run(p, file_config, '202240', 'M')
