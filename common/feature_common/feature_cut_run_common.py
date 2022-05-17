import keras.backend as K
import numpy as np

from config.Config import TrainConfig
from util.load_data import Rectify
from util.load_data import load_data_list
from util.load_data_feature_cut import load_train_or_test_cut_feature_C, load_train_or_test_cut_feature_M
from util.scoreClass import Metric


def feature_cut_run(p, file_config, data_type, model_name, class_type: str, get_and_load_model, feature_cut_list,
                    sort_type) -> None:
    """
    :param p: 项目根目录地址
    :param file_config: 训练文件配置类对象
    :param data_type: 数据集类型  TT TVT 2018 2022
    :param class_type: 分类类型  C  M
    :param model_name: 模型名称
    :param get_and_load_model: 加载模型的方法
    :param feature_cut_list: 去掉的特征名称
    :param sort_type: asc 或desc
    """
    print(feature_cut_list)
    train_config = TrainConfig()
    test_list = load_data_list(file_config.test_file)
    print(file_config.test_file)
    for time_steps in [40]:
        print(model_name + " " + str(len(feature_cut_list)) + " " + str(time_steps))
        all_metric = {
            "TSS": [0, 0]
        }
        all_matrix = np.array([[0, 0], [0, 0]])
        data_Recall, data_Precision, data_Accuracy, data_TSS, data_HSS, data_FAR = [], [], [], [], [], []
        all_nums = 0
        for i in range(len(test_list)):  # 循环10-Fold文件
            all_nums += 1
            x_test, y_test, test_weight_dir = None, None, None
            if class_type == 'C':
                x_test, y_test, test_weight_dir = load_train_or_test_cut_feature_C(test_list[i], feature_cut_list)
            elif class_type == 'M':
                x_test, y_test, test_weight_dir = load_train_or_test_cut_feature_M(test_list[i], feature_cut_list)
            # 载入模型
            # 根据时间步修改测试集shape
            if model_name == 'NN':
                x_test_time_step = Rectify(x_test, time_steps)
                model_path = p + '/weights/feature_impotence_cut/' + data_type + '/' + sort_type + '/NN_best≥' + class_type + '_time_steps=1' + '/' + str(
                    len(feature_cut_list)) + '/' + 'NN_feature_cut_' + str(len(feature_cut_list)) + '_' + str(i) + '.h5'
            else:
                x_test_time_step = x_test.reshape(-1, time_steps, 10 - len(feature_cut_list))
                model_path = p + '/weights/feature_impotence_cut/' + data_type + '/' + sort_type + '/' + model_name + '_best≥' + class_type + '/time_steps=' + str(
                    time_steps) + '/' + str(len(feature_cut_list)) + '/' + model_name + '_feature_cut_' + str(
                    time_steps) + '_' + str(len(feature_cut_list)) + '_' + str(i) + '.h5'
            # print(model_path)
            # print(test_list[i])
            model = get_and_load_model(
                time_steps=time_steps,
                learning_rate=train_config.learning_rate,
                dropout_rate=train_config.dropout_rate,
                seed=train_config.glorot_normal_seed,
                score_metrics=train_config.score_metrics,
                feature_size=10 - len(feature_cut_list)
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
        print(data_TSS[:, 0].mean(), end='\t')
        print(data_TSS[:, 0].std(), end='\t')
        print('%.4f' % data_TSS[:, 0].mean(), end='±')
        print('%.4f' % data_TSS[:, 0].std())
        print("==================================")
        K.clear_session()
