import sys
import os

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import File2018Config
from main.train_common.M.GRU_attention_train_M import GRU_attention_M_train

file_config = File2018Config(p)
GRU_attention_M_train(p, file_config, '2018')
