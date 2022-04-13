import sys
import os

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import File2022Config
from main.train_common.C.Bi_GRU_train_C import Bi_GRU_C_train

file_config = File2022Config(p)
Bi_GRU_C_train(p, file_config, '2022')
