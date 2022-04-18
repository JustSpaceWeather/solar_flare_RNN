import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.GRU_Att_run import GRU_Att
from config.Config import File2022Config

file_config = File2022Config(p)
GRU_Att(p, file_config, '2022', 'M')
