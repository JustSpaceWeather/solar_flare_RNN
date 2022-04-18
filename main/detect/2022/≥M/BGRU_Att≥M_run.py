import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.BGRU_Att_run import BGRU_Att
from config.Config import TTFileConfig

file_config = TTFileConfig(p)
BGRU_Att(p, file_config, 'TVT', 'M')
