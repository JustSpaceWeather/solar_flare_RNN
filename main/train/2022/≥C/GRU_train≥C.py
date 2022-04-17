import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import File2022Config
from common.train_common.C.GRU_train_C import GRU_C_train

file_config = File2022Config(p)
GRU_C_train(p, file_config, '2022')
