import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.BGRU_run import BGRU
from config.Config import TTFileConfig

file_config = TTFileConfig(p)
BGRU(p, file_config, 'TVT', 'C')
