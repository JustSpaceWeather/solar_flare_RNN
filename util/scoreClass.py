import sys

import numpy as np
from sklearn.metrics import confusion_matrix


# sys.float_info.epsilon用在分母上，无实际意义，仅用于防止分母为0报错
class Metric(object):
    def __init__(self, y_true, y_pred):
        self.__matrix = confusion_matrix(y_true, y_pred)

    def Matrix(self):
        return self.__matrix

    def TP(self):
        tp = np.diag(self.__matrix)  # diag 输出二维矩阵对角线元素
        return tp.astype(float)

    def TN(self):
        tn = self.__matrix.sum() - (self.FP() + self.FN() + self.TP())
        return tn.astype(float)

    def FP(self):
        fp = self.__matrix.sum(axis=0) - np.diag(self.__matrix)
        return fp.astype(float)

    def FN(self):
        fn = self.__matrix.sum(axis=1) - np.diag(self.__matrix)
        return fn.astype(float)

    def TPRate(self):  # TPR
        return self.TP() / (self.TP() + self.FN() + sys.float_info.epsilon)

    def TNRate(self):
        return self.TN() / (self.TN() + self.FP() + sys.float_info.epsilon)

    def FPRate(self):  # FPR
        return 1 - self.TNRate()

    def FNRate(self):
        return 1 - self.TPRate()

    def Accuracy(self):
        ALL = self.TP() + self.FP() + self.TN() + self.FN()
        RIGHT = self.TP() + self.TN()
        return RIGHT / (ALL + sys.float_info.epsilon)

    def BACC(self):  # Balanced Accuracy
        return ((self.TP() / (self.TP() + self.FN() + sys.float_info.epsilon)) + (
                    self.TN() / (self.TN() + self.FP() + sys.float_info.epsilon))) / 2.0

    def Recall(self):
        return self.TP() / (self.TP() + self.FN() + sys.float_info.epsilon)

    def Precision(self):
        return self.TP() / (self.TP() + self.FP() + sys.float_info.epsilon)

    def TSS(self):
        return self.TPRate() - self.FPRate()

    def HSS(self):
        P = self.TP() + self.FN()
        N = self.TN() + self.FP()
        up = 2 * (self.TP() * self.TN() - self.FN() * self.FP())
        below = P * (self.FN() + self.TN()) + N * (self.TP() + self.FP())
        return up / (below + sys.float_info.epsilon)

    def FAR(self):  # false alarm ratio 空报率
        return self.FP() / (self.FP() + self.TP() + sys.float_info.epsilon)

    def POD(self):
        return self.TP() / (self.TP() + self.FN() + sys.float_info.epsilon)

    def CSI(self):
        return self.TP() / (self.TP() + self.FP() + self.FN() + sys.float_info.epsilon)
