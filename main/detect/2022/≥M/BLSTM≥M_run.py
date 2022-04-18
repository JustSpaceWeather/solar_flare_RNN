import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.BLSTM_run import BLSTM
from config.Config import File2022Config

file_config = File2022Config(p)
BLSTM(p, file_config, '2022', 'M')
