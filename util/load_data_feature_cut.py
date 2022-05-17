import pandas as pd
from tensorflow.python.keras.utils import np_utils


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


# 以下代码用于特征重要性分析，asc,desc中
def load_train_or_test_cut_feature(filepath, class_1: list, class_2: list, cut_feature_list: list):
    data_list = []  # 用于存放x_train, y_train, x_test, y_test
    csv = pd.read_csv(filepath)
    start = 0  # strat用于获取列名为:TOTUSJH的下标，end用于获取列名为：SHRGT45的下标
    for column in csv.columns.values:  # 找到开始(start)和结束(end)的列数索引
        start += 1
        if column.__eq__("CLASS"):  # 找到TOTUSJH所在列
            break

    pd.set_option('display.max_columns', None)
    csv.drop(csv.columns[range(start - 1)].tolist(), axis=1, inplace=True)
    csv.drop(cut_feature_list, axis=1, inplace=True)
    end = 10 - len(cut_feature_list)
    data_list.append(csv.iloc[:, 1:end + 1].values)

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

    return data_list[0], data_list[1], get_weight_dir([class_1_num, class_2_num])  # x_train, y_train, weight


def load_data_feature_cut(train_path, test_path, class_1, class_2, feature_cut_list):
    return load_train_or_test_cut_feature(train_path, class_1, class_2, feature_cut_list), \
           load_train_or_test_cut_feature(test_path, class_1, class_2, feature_cut_list)


def load_data_cut_feature_C(train_path, test_path, feature_cut_list):
    class_1 = ['N']
    class_2 = ['C', 'M', 'X']
    return load_data_feature_cut(train_path, test_path, class_1, class_2, feature_cut_list)


def load_data_cut_feature_M(train_path, test_path, feature_cut_list):
    class_1 = ['N', 'C']
    class_2 = ['M', 'X']
    return load_data_feature_cut(train_path, test_path, class_1, class_2, feature_cut_list)


def load_train_or_test_cut_feature_C(file_path, feature_cut_list):
    class_1 = ['N']
    class_2 = ['C', 'M', 'X']
    return load_train_or_test_cut_feature(file_path, class_1, class_2, feature_cut_list)


def load_train_or_test_cut_feature_M(file_path, feature_cut_list):
    class_1 = ['N', 'C']
    class_2 = ['M', 'X']
    return load_train_or_test_cut_feature(file_path, class_1, class_2, feature_cut_list)
