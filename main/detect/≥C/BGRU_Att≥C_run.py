import sys
import os
import numpy as np

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(p)

from util.load_data import load_train_or_test_C
from util.load_data import load_data_list
from model.Bi_GRU_attention_model import get_Bi_GRU_attention_model
from util.scoreClass import Metric
from util.load_data import Rectify
from config.Config import DetectConfig
from config.Config import TTFileConfig

config = DetectConfig()
file_config = TTFileConfig(p)
time_steps_list = config.time_steps_list
test_list = load_data_list(file_config.test_file)
score_metrics = config.score_metrics

if __name__ == '__main__':
    for time_steps in time_steps_list:
        all_metric = {
            "Recall": [0, 0],
            "Precision": [0, 0],
            "Accuracy": [0, 0],
            "TSS": [0, 0],
            "HSS": [0, 0],
            "FAR": [0, 0]
        }
        all_matrix = np.array([[0, 0], [0, 0]])
        data_Recall, data_Precision, data_Accuracy, data_TSS, data_HSS, data_FAR = [], [], [], [], [], []
        all_nums = 0
        for i in range(10):
            all_nums += 1
            x_test, y_test, test_weight_dir = load_train_or_test_C(test_list[i])
            # 载入模型
            model = get_Bi_GRU_attention_model(learning_rate=1e-4, dropout_rate=0.0, glorot_normal_seed=369,
                                               score_metrics=score_metrics, time_steps=time_steps)
            model.load_weights(
                p + r'\weights\Bi_GRU_attention_best≥C\time_steps=' + str(time_steps) + r'\Bi_GRU_attention_C_' + str(
                    time_steps
                ) + '_best_' + str(i) + '.h5'
            )
            # 根据时间步修改测试集shape
            x_test_time_step = x_test.reshape(-1, time_steps, 10)
            y_test_time_step = Rectify(y_test, time_steps)
            # 开始预测
            y_true = y_test_time_step.argmax(axis=1)
            y_pred = model.predict(x_test_time_step).argmax(axis=1)
            # 指标输出
            metric = Metric(y_true, y_pred)
            # print("time_steps", time_steps)
            # print("Matrix:\n", metric.Matrix())
            # print("Recall:", metric.Recall())
            # print("Precision:", metric.Precision())
            # print("Accuracy:", metric.Accuracy())
            # print("TSS:", metric.TSS())
            # print("HSS:", metric.HSS())
            # print("FAR:", metric.FAR())
            print(metric.Recall())
            print(metric.Precision())
            print(metric.Accuracy())
            print(metric.TSS())
            print(metric.HSS())
            print(metric.FAR())
            data_Recall.extend(metric.Recall())
            data_Precision.extend(metric.Precision())
            data_Accuracy.extend(metric.Accuracy())
            data_TSS.extend(metric.TSS())
            data_HSS.extend(metric.HSS())
            data_FAR.extend(metric.FAR())
            all_matrix += metric.Matrix()
            all_metric["Recall"] += metric.Recall()
            all_metric["Precision"] += metric.Precision()
            all_metric["Accuracy"] += metric.Accuracy()
            all_metric["TSS"] += metric.TSS()
            all_metric["HSS"] += metric.HSS()
            all_metric["FAR"] += metric.FAR()
            # print('------------------------------------------')
        # print(all_matrix)
        # for index in all_metric:
        #     print(index, np.array(all_metric[index]) / all_nums)
        # print("\n")
        data_TSS = np.array(data_TSS).reshape(10, 2)
        data_Precision = np.array(data_Precision).reshape(10, 2)
        data_Accuracy = np.array(data_Accuracy).reshape(10, 2)
        data_Recall = np.array(data_Recall).reshape(10, 2)
        data_FAR = np.array(data_FAR).reshape(10, 2)
        data_HSS = np.array(data_HSS).reshape(10, 2)
        # print("各种指标的平均值与标准差如下所示：")
        # print("Recall mean:", data_Recall[:, 0].mean(), data_Recall[:, 1].mean())
        # print("Recall std:", data_Recall[:, 0].std(), data_Recall[:, 1].std())
        # print("TSS mean:", data_TSS[:, 0].mean(), data_TSS[:, 1].mean())
        # print("TSS std:", data_TSS[:, 0].std(), data_TSS[:, 1].std())
        # print("HSS mean:", data_HSS[:, 0].mean(), data_HSS[:, 1].mean())
        # print("HSS std:", data_HSS[:, 0].std(), data_HSS[:, 1].std())
        # print("Accuracy mean:", data_Accuracy[:, 0].mean(), data_Accuracy[:, 1].mean())
        # print("Accuracy std:", data_Accuracy[:, 0].std(), data_Accuracy[:, 1].std())
        # print("Precision mean:", data_Precision[:, 0].mean(), data_Precision[:, 1].mean())
        # print("Precision std:", data_Precision[:, 0].std(), data_Precision[:, 1].std())
        # print("FAR mean:", data_FAR[:, 0].mean(), data_FAR[:, 1].mean())
        # print("FAR std:", data_FAR[:, 0].std(), data_FAR[:, 1].std())

        # print(time_steps)
        print(data_Recall[:, 0].mean(), data_Recall[:, 1].mean())
        print(data_Recall[:, 0].std(), data_Recall[:, 1].std())
        print(data_TSS[:, 0].mean(), data_TSS[:, 1].mean())
        print(data_TSS[:, 0].std(), data_TSS[:, 1].std())
        print(data_HSS[:, 0].mean(), data_HSS[:, 1].mean())
        print(data_HSS[:, 0].std(), data_HSS[:, 1].std())
        print(data_Accuracy[:, 0].mean(), data_Accuracy[:, 1].mean())
        print(data_Accuracy[:, 0].std(), data_Accuracy[:, 1].std())
        print(data_Precision[:, 0].mean(), data_Precision[:, 1].mean())
        print(data_Precision[:, 0].std(), data_Precision[:, 1].std())
        print(data_FAR[:, 0].mean(), data_FAR[:, 1].mean())
        print(data_FAR[:, 0].std(), data_FAR[:, 1].std())
        print("==================================")
