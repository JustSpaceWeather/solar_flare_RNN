import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.GRU_run import GRU
from config.Config import TTFileConfig

file_config = TTFileConfig(p)
GRU(p, file_config, '2018', 'M')
