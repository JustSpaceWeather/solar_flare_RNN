import numpy as np
from keras import backend as K

from config.Config import DetectConfig
from config.Config import TrainConfig
from util.get_model_path import get_model_path
from util.load_data import Rectify
from util.load_data import load_data_list
from util.load_data import load_train_or_test_C, load_train_or_test_M
from util.score import BS_BSS_score
from util.scoreClass import Metric

data_dir_40 = {
    1: 40,
    2: 20,
    5: 8,
    10: 4,
    20: 2,
    40: 1
}

data_dir_120 = {  # 暂时用不到，还没编写

}


def changShape(data, time_steps):
    temp_data = []
    for i in range(0, data.shape[0], data_dir_40[time_steps]):
        temp_data.append(data[i])
    data = np.array(temp_data)
    return data


def detect_same(p, file_config, detect_type, class_type: str, model_name, get_model) -> None:
    """
    :param p: 项目根目录地址
    :param file_config: 训练文件配置类对象
    :param detect_type: 数据集类型  TT TVT 2018 2022
    :param class_type: 分类类型  C  M
    :param model_name: 模型名称
    :param get_model: 加载模型的方法
    """
    detect_config = DetectConfig()
    train_config = TrainConfig()
    test_list = load_data_list(file_config.test_file)
    if model_name != 'NN':
        time_steps_list = detect_config.time_steps_list
    else:
        time_steps_list = [40]
    for time_steps in time_steps_list:
        print(time_steps, model_name, file_config.test_file)
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
        BS_all = []
        BSS_all = []
        all_nums = 0
        for i in range(10):  # 循环10-Fold文件
            all_nums += 1
            x_test, y_test, test_weight_dir = None, None, None
            if class_type == 'C':
                x_test, y_test, test_weight_dir = load_train_or_test_C(test_list[i])
            elif class_type == 'M':
                x_test, y_test, test_weight_dir = load_train_or_test_M(test_list[i])
            # 载入模型
            # 根据时间步修改测试集shape
            if model_name == 'NN':
                x_test_time_step = Rectify(x_test, time_steps)
                model_path = get_model_path(p, detect_type, class_type, model_name, 1, i)
            else:
                x_test_time_step = x_test.reshape(-1, time_steps, 10)
                model_path = get_model_path(p, detect_type, class_type, model_name, time_steps, i)
            model = get_model(
                time_steps=time_steps,
                learning_rate=train_config.learning_rate,
                dropout_rate=0.0,
                seed=train_config.glorot_normal_seed,
                score_metrics=train_config.score_metrics
            )
            model.load_weights(model_path)
            y_test_time_step = Rectify(y_test, time_steps)
            # 开始预测
            # 开始预测
            y_true = y_test_time_step.argmax(axis=1)
            y_prob = model.predict(x_test_time_step)
            y_pred = y_prob.argmax(axis=1)
            y_prob = y_prob[:, 1]
            y_true = changShape(y_true, time_steps)
            y_pred = changShape(y_pred, time_steps)
            y_prob = changShape(y_prob, time_steps)
            # print(y_true)
            # print(y_prob)
            # print(y_pred)
            # BSS开始计算
            BS, BSS = BS_BSS_score(y_true, y_prob)
            # 指标输出
            metric = Metric(y_true, y_pred)
            # print(metric.Matrix())
            print(metric.Matrix()[0][0], end='\t')
            print(metric.Matrix()[0][1])
            print(metric.Matrix()[1][0], end='\t')
            print(metric.Matrix()[1][1])
            # print(metric.Recall()[0])
            # print(metric.Precision()[0])
            # print(metric.Accuracy()[0])
            print(metric.TSS()[0])
            # print(metric.HSS()[0])
            # print(metric.FAR()[0])
            # print(BS)
            # print(BSS)
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
            BS_all.append(BS)
            BSS_all.append(BSS)
        # print(all_matrix)
        print(all_matrix[0][0],end='\t')
        print(all_matrix[0][1])
        print(all_matrix[1][0],end='\t')
        print(all_matrix[1][1])
        data_TSS = np.array(data_TSS).reshape(10, 2)
        data_Precision = np.array(data_Precision).reshape(10, 2)
        data_Accuracy = np.array(data_Accuracy).reshape(10, 2)
        data_Recall = np.array(data_Recall).reshape(10, 2)
        data_FAR = np.array(data_FAR).reshape(10, 2)
        data_HSS = np.array(data_HSS).reshape(10, 2)
        BS_all = np.array(BS_all)
        BSS_all = np.array(BSS_all)
        # ####################
        # print('recall')
        # print(data_Recall[:, 0].mean())
        # print(data_Recall[:, 0].std())
        # print('%.4f' % data_Recall[:, 0].mean(), end='±')
        # print('%.4f' % data_Recall[:, 0].std())
        ####################
        print('tss')
        # print(data_TSS[:, 0].mean())
        # print(data_TSS[:, 0].std())
        print('%.4f' % data_TSS[:, 0].mean(), end='±')
        print('%.4f' % data_TSS[:, 0].std())
        ###################
        # print('hss')
        # print(data_HSS[:, 0].mean())
        # print(data_HSS[:, 0].std())
        # print('%.4f' % data_HSS[:, 0].mean(), end='±')
        # print('%.4f' % data_HSS[:, 0].std())
        # ###################
        # print('acc')
        # print(data_Accuracy[:, 0].mean())
        # print(data_Accuracy[:, 0].std())
        # print('%.4f' % data_Accuracy[:, 0].mean(), end='±')
        # print('%.4f' % data_Accuracy[:, 0].std())
        # ###################
        # print('precision')
        # print(data_Precision[:, 0].mean())
        # print(data_Precision[:, 0].std())
        # print('%.4f' % data_Precision[:, 0].mean(), end='±')
        # print('%.4f' % data_Precision[:, 0].std())
        # ###################
        # print('far')
        # print(data_FAR[:, 0].mean())
        # print(data_FAR[:, 0].std())
        # print('%.4f' % data_FAR[:, 0].mean(), end='±')
        # print('%.4f' % data_FAR[:, 0].std())
        # #########################
        # print('bs', 'bss')
        # print(BS_all.mean())
        # print(BS_all.std())
        # print('%.4f' % BS_all.mean(), end='±')
        # print('%.4f' % BS_all.std(), end='\t')
        # ########################
        # print('bss')
        # print(BSS_all.mean())
        # print(BSS_all.std())
        # print('%.4f' % BSS_all.mean(), end='±')
        # print('%.4f' % BSS_all.std())
        print("==================================")
        K.clear_session()
