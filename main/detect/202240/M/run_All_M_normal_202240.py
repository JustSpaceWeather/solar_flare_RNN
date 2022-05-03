import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.run import *

from config.Config import File202240Config

file_config = File202240Config(p)
# NN(p, file_config, '202240', 'M')

# LSTM(p, file_config, '202240', 'M')
# LSTM_Att(p, file_config, '202240', 'M')
LSTM_Att2(p, file_config, '202240', 'M')

# BLSTM(p, file_config, '202240', 'M')
# BLSTM_Att(p, file_config, '202240', 'M')
BLSTM_Att2(p, file_config, '202240', 'M')

# GRU(p, file_config, '202240', 'M')
# GRU_Att(p, file_config, '202240', 'M')
GRU_Att2(p, file_config, '202240', 'M')

# BGRU(p, file_config, '202240', 'M')
# BGRU_Att(p, file_config, '202240', 'M')
BGRU_Att2(p, file_config, '202240', 'M')
