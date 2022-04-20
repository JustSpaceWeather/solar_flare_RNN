import os
import random

import numpy as np
import tensorflow as tf
from tfdeterminism import patch

from config.Config import SeedConfig


def set_seed():
    config = SeedConfig()
    # 下方代码解决训练结果可复现的问题，设置随机数种子
    # 模型结果可复现解决方案：https://zhuanlan.zhihu.com/p/95416326
    os.environ['PYTHONHASHSEED'] = config.PYTHONHASHSEED
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    patch()
    tf.set_random_seed(config.tf_random_seed)



# warnings.filterwarnings("ignore")  # 不显示警告日志输出
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error
