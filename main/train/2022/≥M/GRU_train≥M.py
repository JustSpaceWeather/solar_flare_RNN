import sys
import os

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from config.Config import File2022Config
from main.train_common.M.GRU_train_M import GRU_M_train

file_config = File2022Config(p)
GRU_M_train(p, file_config, '2022')
