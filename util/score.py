from keras import Model
from keras import backend as K
from sklearn.metrics import brier_score_loss

from util.scoreClass import Metric


def BSS(y_true, y_pred):
    BS = K.mean(K.square(y_true - y_pred), axis=0)[1]  # 每一行的差的平方的平均值
    y_ave = K.mean(y_true, axis=0)
    return 1 - BS / (K.mean(K.square(y_true - y_ave), axis=0)[1] + K.epsilon())


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
