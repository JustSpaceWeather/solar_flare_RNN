import matplotlib.pyplot as plt
import numpy as np

from config.Enum import *

"""
蓝色条代表平均个人TSS分数，红色折线代表特征的平均累积TSS分数。
"""

NN_TSS_asc = [0.6664, 0.5893, 0.5707, 0.5621, 0.5521, 0.5357, 0.5007, 0.4464, 0.3536, 0.2971]
LSTM_TSS_asc = [0.7329, 0.6671, 0.6436, 0.6221, 0.605, 0.595, 0.5876, 0.4886, 0.4143, 0.3243]
LSTM_Att_TSS_asc = [0.73, 0.66, 0.6514, 0.6307, 0.615, 0.5864, 0.5829, 0.4979, 0.4164, 0.3357]
BLSTM_TSS_asc = [0.7314, 0.6614, 0.6436, 0.6286, 0.6079, 0.595, 0.5829, 0.4943, 0.4143, 0.3314]
BLSTM_Att_TSS_asc = [0.7307, 0.6664, 0.6529, 0.6229, 0.61, 0.59, 0.5857, 0.4929, 0.4929, 0.3421]
GRU_TSS_asc = [0.7229, 0.6479, 0.6386, 0.6271, 0.6057, 0.595, 0.5814, 0.4821, 0.4243, 0.3293]
GRU_Att_TSS_asc = [0.7164, 0.665, 0.655, 0.6121, 0.6, 0.5857, 0.5771, 0.4821, 0.41, 0.3371]
BGRU_TSS_asc = [0.7329, 0.6636, 0.6407, 0.6214, 0.6029, 0.5857, 0.5757, 0.4971, 0.4114, 0.335]
BGRU_Att_TSS_asc = [0.7243, 0.665, 0.66, 0.6121, 0.5971, 0.5857, 0.5814, 0.4857, 0.4171, 0.3329]

NN_std_asc = [0.0511, 0.0537, 0.0636, 0.0597, 0.0707, 0.0636, 0.0721, 0.0663, 0.0492, 0.0524]
LSTM_std_asc = [0.0382, 0.0459, 0.0682, 0.0551, 0.0529, 0.0579, 0.0561, 0.0597, 0.0505, 0.0477]
LSTM_Att_std_asc = [0.0364, 0.0461, 0.0504, 0.0508, 0.0672, 0.0552, 0.0596, 0.0591, 0.047, 0.0416]
BLSTM_std_asc = [0.037, 0.0504, 0.0594, 0.053, 0.0683, 0.0605, 0.0528, 0.0563, 0.0528, 0.0474]
BLSTM_Att_std_asc = [0.0368, 0.0486, 0.068, 0.0519, 0.0619, 0.0574, 0.0581, 0.0518, 0.0518, 0.0434]
GRU_std_asc = [0.0403, 0.0341, 0.0679, 0.0531, 0.0606, 0.0541, 0.0547, 0.0454, 0.0572, 0.0526]
GRU_Att_std_asc = [0.0459, 0.0407, 0.0607, 0.0583, 0.0677, 0.0584, 0.0648, 0.0486, 0.0371, 0.0388]
BGRU_std_asc = [0.0401, 0.0465, 0.0637, 0.0586, 0.0641, 0.0564, 0.0623, 0.0612, 0.0362, 0.0495]
BGRU_Att_std_asc = [0.0366, 0.0469, 0.0549, 0.0595, 0.061, 0.058, 0.0612, 0.0474, 0.0507, 0.0381]

all_title_list = [
    'NN', 'LSTM', 'LSTM_Att', 'BLSTM', 'BLSTM_Att', 'GRU', 'GRU_Att', 'BGRU', 'BGRU_Att'
]

all_name_list = [
    feature_M_asc['NN'],
    feature_M_asc['LSTM'],
    feature_M_asc['LSTM_Att2'],
    feature_M_asc['BLSTM'],
    feature_M_asc['BLSTM_Att2'],
    feature_M_asc['GRU'],
    feature_M_asc['GRU_Att2'],
    feature_M_asc['BGRU'],
    feature_M_asc['BGRU_Att2']
]
all_TSS_list = [
    NN_TSS_asc,
    LSTM_TSS_asc,
    LSTM_Att_TSS_asc,
    BLSTM_TSS_asc,
    BLSTM_Att_TSS_asc,
    GRU_TSS_asc,
    GRU_Att_TSS_asc,
    BGRU_TSS_asc,
    BGRU_Att_TSS_asc
]
all_std_list = [
    NN_std_asc,
    LSTM_std_asc,
    LSTM_Att_std_asc,
    BLSTM_std_asc,
    BLSTM_Att_std_asc,
    GRU_std_asc,
    GRU_Att_std_asc,
    BGRU_std_asc,
    BGRU_Att_std_asc
]

for i in range(len(all_std_list)):
    plt.figure(dpi=400)
    # 设置x轴坐标名称
    names = [""]  # x轴坐标名称
    names = names + all_name_list[i]
    name_range = range(len(names))  # x轴坐标名称对于的range索引
    x = np.array([i for i in range(1, 11)])  # x轴坐标画图位置
    y = np.array(all_TSS_list[i])  # 个人TSS分数
    x_err = np.array(all_std_list[i])  # 个人TSS分数标准差

    # 开始画图
    plt.title(all_title_list[i] + '≥M')
    plt.ylim((0, 0.8))
    plt.xlim((0, 11))
    plt.xticks(name_range, names, rotation=90)  #
    plt.bar(x, y, yerr=x_err, ecolor='black')
    plt.tight_layout()  # 调整画布大小，防止显示不全
    plt.savefig('D:/M_only/' + str(i) + '_M_only_' + all_title_list[i] + '.jpg')
    plt.show()

# 调整子图形之间的纵向距离
# figure.subplots_adjust(hspace=0.1)

# plt.errorbar(x, y, yerr=x_err, ecolor='black', color='red', linestyle='--')
