import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import File2018Config
from common.train_common.C.GRU_attention2_train_C import GRU_attention2_C_train

file_config = File2018Config(p)
GRU_attention2_C_train(p, file_config, '2018')
