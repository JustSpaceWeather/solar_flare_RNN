import os
import sys

from common.detect_common.run import *

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import File202240Config

file_config = File202240Config(p)
NN(p, file_config, '202240', 'C')
#
LSTM(p, file_config, '202240', 'C')
# LSTM_Att(p, file_config, '202240', 'C')
LSTM_Att2(p, file_config, '202240', 'C')

BLSTM(p, file_config, '202240', 'C')
# BLSTM_Att(p, file_config, '202240', 'C')
BLSTM_Att2(p, file_config, '202240', 'C')

GRU(p, file_config, '202240', 'C')
# GRU_Att(p, file_config, '202240', 'C')
GRU_Att2(p, file_config, '202240', 'C')

BGRU(p, file_config, '202240', 'C')
# BGRU_Att(p, file_config, '202240', 'C')
BGRU_Att2(p, file_config, '202240', 'C')
