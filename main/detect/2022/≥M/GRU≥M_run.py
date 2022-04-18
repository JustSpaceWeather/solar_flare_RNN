import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.GRU_run import GRU
from config.Config import File2022Config

file_config = File2022Config(p)
GRU(p, file_config, '2022', 'M')
