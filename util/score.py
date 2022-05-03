import numpy as np
from keras import Model
from sklearn.metrics import brier_score_loss

from util.scoreClass import Metric


def BS_BSS_score(y_true, y_prob):
    """
    :param y_true: one_hot格式
    :param y_prob: softmax输出的(m, 2)形状的
    :return: BS和BSS的值
    y_true = y_test.argmax(axis=1)
    y_prob = model.predict(x_test_time_step)[:, 1]
    """
    # BSS开始计算
    BS = brier_score_loss(y_true, y_prob)
    y_mean = y_prob.mean()
    # print(y_true)
    # print(y_mean)
    temp = y_true - y_mean
    # print(temp)
    temp = np.square(temp)
    # print(temp)
    temp = np.sum(temp) / float(len(y_true))
    BSS = 1 - BS / temp
    return BS, BSS


def show_score_and_save_weights(model: Model, best_TSS, y_true, y_pred, filename) -> float:
    """
    显示当前权重的得分，并且保存比best_TSS更好的TSS值的模型的权重
    :param model: 传入含有权重的模型
    :param best_TSS: 训练过程中已经取得的最好的TSS的值
    :param y_true: 真实的值，one_hot形式
    :param y_pred: 预测的值，one_hot形式
    :param filename: 权重保存的位置
    :return: 如果当前模型的TSS值比best_TSS的值大，则返回当前模型的TSS值，否则返回传入的best_TSS
    """
    metric = Metric(y_true, y_pred)  # 计算当前模型的TSS值
    new_TSS = metric.TSS()[0]
    print('Recall', metric.Recall(),
          '\nHSS', metric.HSS(),
          '\nAccuracy', metric.Accuracy(),
          '\nPrecision', metric.Precision(),
          '\nFAR', metric.FAR(),
          '\nTSS =', new_TSS,
          '\nbest_TSS=', best_TSS)
    if new_TSS > best_TSS:
        print("TSS从{0}提升至{1}".format(best_TSS, new_TSS))
        best_TSS = new_TSS
        model.save_weights(filename)
    return best_TSS


