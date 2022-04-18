import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.BGRU_run import BGRU
from config.Config import File2018Config

file_config = File2018Config(p)
BGRU(p, file_config, '2018', 'C')
