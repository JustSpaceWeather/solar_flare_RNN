import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import TTFileConfig
from common.train_common.C.GRU_attention2_train_C import GRU_attention2_C_train

file_config = TTFileConfig(p)
GRU_attention2_C_train(p, file_config, 'TT')
