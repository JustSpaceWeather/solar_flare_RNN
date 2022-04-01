import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from util.scoreClass import Metric


def base(y_true, y_pred):
    y_pred_positive = K.round(K.clip(y_pred, 0, 1))
    y_pred_negative = 1 - y_pred_positive

    y_positive = K.round(K.clip(y_true, 0, 1))
    y_negative = 1 - y_positive

    TP = K.sum(y_positive * y_pred_positive)
    TN = K.sum(y_negative * y_pred_negative)

    FP = K.sum(y_negative * y_pred_positive)
    FN = K.sum(y_positive * y_pred_negative)

    return TP, TN, FP, FN


def TPRate(y_true, y_pred):
    TP, TN, FP, FN = base(y_true, y_pred)
    return TP / (TP + FN + K.epsilon())


def TNRate(y_true, y_pred):
    TP, TN, FP, FN = base(y_true, y_pred)
    return TN / (TN + FP + K.epsilon())


def FPRate(y_true, y_pred):
    return 1 - TNRate(y_true, y_pred)


def FNRate(y_true, y_pred):
    return 1 - TPRate(y_true, y_pred)


def Accuracy(y_true, y_pred):
    TP, TN, FP, FN = base(y_true, y_pred)
    ALL = TP + FP + TN + FN
    RIGHT = TP + TN
    return RIGHT / (ALL + K.epsilon())


def Recall(y_true, y_pred):
    TP, TN, FP, FN = base(y_true, y_pred)
    return TP / (TP + FN + K.epsilon())


def Precision(y_true, y_pred):
    TP, TN, FP, FN = base(y_true, y_pred)
    return TP / (TP + FP + K.epsilon())


def TSS(y_true, y_pred):
    return TPRate(y_true, y_pred) - FPRate(y_true, y_pred)


def HSS(y_true, y_pred):
    TP, TN, FP, FN = base(y_true, y_pred)
    P = TP + FN
    N = TN + FP
    up = 2 * (TP * TN - FN * FP)
    below = P * (FN + TN) + N * (TP + FP)
    return up / (below + K.epsilon())


def FAR(y_true, y_pred):
    TP, TN, FP, FN = base(y_true, y_pred)
    return FP / (FP + TP + K.epsilon())


def BSS(y_true, y_pred):
    BS = K.mean(K.square(y_true - y_pred), axis=0)[1]
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
