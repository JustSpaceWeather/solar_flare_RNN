import sys
import os

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import TVTFileConfig
from common.train_common.M.GRU_train_M import GRU_M_train

file_config = TVTFileConfig(p)
GRU_M_train(p, file_config, 'TVT')
