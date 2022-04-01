import os

import pandas as pd
import numpy as np

from tensorflow.python.keras.utils import np_utils


def load_data_list(dir_path):
    """
    获取训练集/测试集文件夹下所有的csv文件
    :param dir_path: 数据所在的文件夹
    :return: csv的绝对路径列表，顺序为[1,10,2,3,4,5,6,7,8,9]
    """
    data_list = []
    files = os.listdir(dir_path)
    for i in range(len(files)):
        if files[i].endswith('.csv'):
            files[i] = dir_path + '\\' + files[i]
            data_list.append(files[i])
    return data_list


def load_data(train_path, test_path, class_1: list, class_2: list):
    """
    :train_path:训练数据文件的位置
    :test_path:测试数据文件的位置
    用于读取和生成训练数据和测试数据及对应数据的标签，数据标签为one_hot形式
    :return:返回训练集和测试集x_train, y_train, x_test, y_test
    """
    return load_train_or_test(train_path, class_1, class_2), load_train_or_test(test_path, class_1, class_2)


def load_train_or_test(filepath, class_1: list, class_2: list):
    """
    只载入一个，训练或测试的数据，此代码只用于二分类
    :param filepath:训练或测试数据csv的地址
    :param class_1:类别1
    :param class_2:类别2
    :return: 数据0，标签1，权重{0: ,1: }
    """
    data_list = []  # 用于存放x_train, y_train, x_test, y_test
    csv = pd.read_csv(filepath)
    start, end = 0, 0  # strat用于获取列名为:TOTUSJH的下标，end用于获取列名为：SHRGT45的下标
    for column in csv.columns.values:  # 找到开始(start)和结束(end)的列数索引
        start += 1
        if column.__eq__("TOTUSJH"):  # 找到TOTUSJH所在列
            break
    for column in csv.columns.values:
        end += 1
        if column.__eq__("SHRGT45"):  # 找到SHRGT45所在列
            break
    data_list.append(csv.iloc[:, start - 1:end].values)  # pandas取行操作
    classes = csv['CLASS'].copy()  # 获取csv中CLASS列的所有内容及其索引(从0开始)
    class_list = []
    class_1_num = 0
    class_2_num = 0
    for one_row_class in classes:
        if class_1.count(one_row_class):  # 如果one_row_class是class_1中的，class_list拼接0
            class_list.append(0)
            class_1_num += 1
        if class_2.count(one_row_class):  # 如果one_row_class是class_2中的，class_list拼接1
            class_list.append(1)
            class_2_num += 1
    data_list.append(np_utils.to_categorical(class_list, num_classes=2))  # 将class_list转化为one_hot形式

    return data_list[0], data_list[1], get_weight_dir([class_1_num, class_2_num])


def load_train_or_test_C(path):
    """
    载入以[N]为一类，[C,M,X]为另一类的二分类数据和标签
    :param path: 数据csv文件的地址
    :return:数据0，标签1
    """
    class_1 = ['N']
    class_2 = ['C', 'M', 'X']
    return load_train_or_test(path, class_1, class_2)


def load_train_or_test_M(path):
    """
    载入以[N,C]为一类，[M,X]为另一类的二分类数据和标签
    :param path: 数据csv文件的地址
    :return:数据0，标签1
    """
    class_1 = ['N', 'C']
    class_2 = ['M', 'X']
    return load_train_or_test(path, class_1, class_2)


def load_data_C(train_path, test_path):
    class_1 = ['N']
    class_2 = ['C', 'M', 'X']
    return load_data(train_path, test_path, class_1, class_2)


def load_data_M(train_path, test_path):
    class_1 = ['N', 'C']
    class_2 = ['M', 'X']
    return load_data(train_path, test_path, class_1, class_2)


def get_weight_dir(every_class_num_list: list):
    """
    根据每个类样本数获取每个类的权重
    :param every_class_num_list:每个元素在类中的个数
    :return:返回的对应元素的权重列表（和传入列表顺序一致）
    """
    all_samples = 0
    num_classes = len(every_class_num_list)
    for i in every_class_num_list:
        all_samples += i
    weight_dir = {}
    index = 0
    for i in every_class_num_list:
        weight_dir[index] = all_samples / (i * num_classes)
        index += 1
    return weight_dir


def Rectify(_y, time_steps):
    temp_y = []
    for i in range(0, _y.shape[0], time_steps):
        temp_y.append(_y[i])
    _y = np.array(temp_y)
    return _y

