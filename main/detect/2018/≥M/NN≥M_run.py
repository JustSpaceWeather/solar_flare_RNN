import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)

from common.detect_common.NN_run import NN
from config.Config import File2018Config

file_config = File2018Config(p)
NN(p, file_config, '2018', 'M')
