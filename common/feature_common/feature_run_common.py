import keras.backend as K
import numpy as np

from config.Config import DetectConfig
from config.Config import TrainConfig
from util.get_model_path import get_one_feature_model_path
from util.load_data import Rectify
from util.load_data import load_data_list
from util.load_data import load_train_or_test_C, load_train_or_test_M
from util.scoreClass import Metric


def feature_run(p, file_config, data_type, class_type: str, model_name, get_and_load_model, feature_name) -> None:
    """
    :param p: 项目根目录地址
    :param file_config: 训练文件配置类对象
    :param data_type: 数据集类型  TT TVT 2018 2022
    :param class_type: 分类类型  C  M
    :param model_name: 模型名称
    :param get_and_load_model: 加载模型的方法
    :param feature_name: 特征名称
    """
    detect_config = DetectConfig()
    train_config = TrainConfig()
    test_list = load_data_list(file_config.test_file)
    for time_steps in detect_config.time_steps_list:
        print(model_name + " " + feature_name)
        all_metric = {
            "TSS": [0, 0]
        }
        all_matrix = np.array([[0, 0], [0, 0]])
        data_Recall, data_Precision, data_Accuracy, data_TSS, data_HSS, data_FAR = [], [], [], [], [], []
        all_nums = 0
        for i in range(len(test_list)):  # 循环10-Fold文件
            # print(test_list[i])
            all_nums += 1
            x_test, y_test, test_weight_dir = None, None, None
            if class_type == 'C':
                x_test, y_test, test_weight_dir = load_train_or_test_C(test_list[i], feature_name)
            elif class_type == 'M':
                x_test, y_test, test_weight_dir = load_train_or_test_M(test_list[i], feature_name)
            # 载入模型
            # 根据时间步修改测试集shape
            if model_name == 'NN':
                x_test_time_step = Rectify(x_test, time_steps)
                model_path = get_one_feature_model_path(p, data_type, class_type, model_name, 1, i, feature_name)
            else:
                x_test_time_step = x_test.reshape(-1, time_steps, 1)
                model_path = get_one_feature_model_path(p, data_type, class_type, model_name, time_steps, i,
                                                        feature_name)
            model = get_and_load_model(
                time_steps=time_steps,
                learning_rate=train_config.learning_rate,
                dropout_rate=train_config.dropout_rate,
                seed=train_config.glorot_normal_seed,
                score_metrics=train_config.score_metrics,
                feature_size=1
            )
            model.load_weights(model_path)
            y_test_time_step = Rectify(y_test, time_steps)
            # 开始预测
            y_true = y_test_time_step.argmax(axis=1)
            y_pred = model.predict(x_test_time_step).argmax(axis=1)
            # 指标输出
            metric = Metric(y_true, y_pred)
            # print("Matrix:/n", metric.Matrix())
            print(metric.TSS())
            data_TSS.extend(metric.TSS())
            all_matrix += metric.Matrix()
            all_metric["TSS"] += metric.TSS()
        data_TSS = np.array(data_TSS).reshape(10, 2)
        print(data_TSS[:, 0].mean(), data_TSS[:, 1].mean())
        print(data_TSS[:, 0].std(), data_TSS[:, 1].std())
        print("==================================")
        K.clear_session()
