import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import TTFileConfig
from common.train_common.M.GRU_attention2_train_M import GRU_attention2_M_train

file_config = TTFileConfig(p)
GRU_attention2_M_train(p, file_config, 'TT')
