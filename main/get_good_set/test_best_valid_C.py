import os
import sys

import numpy as np

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)

from util.load_data import load_train_or_test_C
from model.LSTM_model import get_LSTM_model
from util.scoreClass import Metric
from util.load_data import Rectify
from config.Config import DetectConfig
from config.Config import TVTFileConfig

config = DetectConfig()
file_config = TVTFileConfig(p)
time_steps_list = config.time_steps_list
score_metrics = config.score_metrics
test_path = file_config.test_file + '/test_'

if __name__ == '__main__':
    for n in range(1, 5, 1):
        path = 'D:/data/' + str(n) + '/C'
        all_metric = {
            "TSS": [0, 0],
        }
        all_matrix = np.array([[0, 0], [0, 0]])
        data_TSS = []
        all_nums = 0
        for i in range(1, 2, 1):
            all_nums += 1
            x_test, y_test, test_weight_dir = load_train_or_test_C(file_config.test_file + '/test_' + str(i) + '.csv')
            # 载入模型
            model = get_LSTM_model(time_steps=120, learning_rate=1e-3, dropout_rate=0.0, seed=369,
                                   score_metrics=score_metrics)
            model.load_weights(path + str(i) + '.h5')
            x_test_time_step = x_test.reshape(-1, 120, 10)
            y_test_time_step = Rectify(y_test, 120)
            # 开始预测
            y_true = y_test_time_step.argmax(axis=1)
            y_pred = model.predict(x_test_time_step).argmax(axis=1)
            # 指标输出
            metric = Metric(y_true, y_pred)
            print(metric.TSS())
            data_TSS.extend(metric.TSS())
            all_metric["TSS"] += metric.TSS()

        # data_TSS = np.array(data_TSS).reshape(10, 2)
        # print(data_TSS[:, 0].mean(), data_TSS[:, 1].mean())
        # print(data_TSS[:, 0].std(), data_TSS[:, 1].std())
        # print("==================================")
