import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.BLSTM_run import BLSTM
from config.Config import TVTFileConfig

file_config = TVTFileConfig(p)
BLSTM(p, file_config, 'TVT', 'C')
