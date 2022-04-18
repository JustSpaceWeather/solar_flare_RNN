import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.BLSTM_run import BLSTM
from config.Config import File2018Config

file_config = File2018Config(p)
BLSTM(p, file_config, '2018', 'C')
