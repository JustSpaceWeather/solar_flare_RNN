import numpy as np
import keras.backend as K
from util.load_data import load_train_or_test_C, load_train_or_test_M
from util.load_data import load_data_list
from model.Bi_GRU_model import get_Bi_GRU_model
from util.scoreClass import Metric
from util.load_data import Rectify
from config.Config import DetectConfig, TrainConfig


def BGRU_C(p, file_config, detect_type, class_type: str) -> None:
    """
    :param p: 项目根目录地址
    :param file_config: 训练文件配置类对象
    :param detect_type: 数据集类型  TT TVT 2018 2022
    :param class_type: 分类类型  C  M
    """
    detect_config = DetectConfig()
    train_config = TrainConfig()
    test_list = load_data_list(file_config.test_file)
    for time_steps in detect_config.time_steps_list:
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
            # print(test_list[i])
            all_nums += 1
            x_test, y_test, test_weight_dir = None, None, None
            if class_type == 'C':
                x_test, y_test, test_weight_dir = load_train_or_test_C(test_list[i])
            elif class_type == 'M':
                x_test, y_test, test_weight_dir = load_train_or_test_M(test_list[i])
            # 载入模型
            model = get_Bi_GRU_model(
                time_steps=time_steps,
                learning_rate=train_config.learning_rate,
                dropout_rate=0.0,
                seed=train_config.glorot_normal_seed,
                score_metrics=detect_config.score_metrics
            )
            model.load_weights(
                p + '/weights/' + detect_type + '/Bi_GRU_best≥' + class_type + '/time_steps=' + str(
                    time_steps) + '/Bi_GRU_' + class_type + '_' + str(time_steps) + '_best_' + str(i) + '.h5'
            )
            # 根据时间步修改测试集shape
            x_test_time_step = x_test.reshape(-1, time_steps, 10)
            y_test_time_step = Rectify(y_test, time_steps)
            # 开始预测
            y_true = y_test_time_step.argmax(axis=1)
            y_pred = model.predict(x_test_time_step).argmax(axis=1)
            # 指标输出
            metric = Metric(y_true, y_pred)
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
        data_TSS = np.array(data_TSS).reshape(10, 2)
        data_Precision = np.array(data_Precision).reshape(10, 2)
        data_Accuracy = np.array(data_Accuracy).reshape(10, 2)
        data_Recall = np.array(data_Recall).reshape(10, 2)
        data_FAR = np.array(data_FAR).reshape(10, 2)
        data_HSS = np.array(data_HSS).reshape(10, 2)
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
        K.clear_session()