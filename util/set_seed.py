import os
import random
import numpy as np
import tensorflow as tf
from tfdeterminism import patch


def set_seed():
    # 下方代码解决训练结果可复现的问题，设置随机数种子
    # 模型结果可复现解决方案：https://zhuanlan.zhihu.com/p/95416326
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(147)
    np.random.seed(258)
    patch()
    tf.set_random_seed(110)
