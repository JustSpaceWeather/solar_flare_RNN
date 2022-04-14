import sys
import os

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import TVTFileConfig
from common.train_common.M.Bi_GRU_attention_train_M import Bi_GRU_attention_M_train

file_config = TVTFileConfig(p)
Bi_GRU_attention_M_train(p, file_config, 'TVT')
