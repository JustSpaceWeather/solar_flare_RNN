import sys
import os

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import TTFileConfig
from main.train_common.C.Bi_GRU_attention_train_C import Bi_GRU_attention_C_train

file_config = TTFileConfig(p)
Bi_GRU_attention_C_train(p, file_config, 'TT')
