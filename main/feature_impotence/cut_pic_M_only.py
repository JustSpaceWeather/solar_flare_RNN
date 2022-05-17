import matplotlib.pyplot as plt
import numpy as np

from config.Enum import *

"""
蓝色条代表平均个人TSS分数，红色折线代表特征的平均累积TSS分数。
"""
figure_cut_M_asc = feature_M_asc

NN_TSS_asc = [0.542, 0.4861, 0.4738, 0.4457, 0.4253, 0.4181, 0.4157, 0.4049, 0.2824, 0.2748]
LSTM_TSS_asc = [0.6667, 0.6369, 0.6166, 0.5998, 0.5269, 0.5226, 0.4861, 0.4462, 0.3475, 0.3324]
LSTM_Att_TSS_asc = [0.6633, 0.5987, 0.597, 0.5888, 0.5483, 0.5434, 0.482, 0.4405, 0.3461, 0.3331]
BLSTM_TSS_asc = [0.6594, 0.6353, 0.6078, 0.5891, 0.5324, 0.5259, 0.4836, 0.4461, 0.3512, 0.3418]
BLSTM_Att_TSS_asc = [0.6655, 0.5958, 0.5941, 0.5824, 0.5642, 0.5551, 0.4726, 0.4495, 0.3451, 0.3305]
GRU_TSS_asc = [0.6631, 0.5907, 0.5783, 0.5613, 0.5261, 0.5132, 0.4793, 0.4375, 0.3389, 0.3255]
GRU_Att_TSS_asc = [0.6633, 0.6055, 0.5909, 0.5634, 0.5631, 0.5437, 0.4607, 0.4388, 0.3574, 0.3399]
BGRU_TSS_asc = [0.6641, 0.5977, 0.5976, 0.5822, 0.529, 0.5219, 0.4765, 0.4409, 0.3415, 0.3278]
BGRU_Att_TSS_asc = [0.6646, 0.5973, 0.5879, 0.5866, 0.5732, 0.5555, 0.4716, 0.4488, 0.3442, 0.3126]

NN_std_asc = [0.0374, 0.0497, 0.0533, 0.0644, 0.0579, 0.0673, 0.0351, 0.0424, 0.053, 0.0654]
LSTM_std_asc = [0.0509, 0.0459, 0.0502, 0.0499, 0.0491, 0.0358, 0.0467, 0.0339, 0.0557, 0.0578]
LSTM_Att_std_asc = [0.0536, 0.0517, 0.0386, 0.0492, 0.0425, 0.0303, 0.046, 0.0317, 0.0523, 0.0572]
BLSTM_std_asc = [0.0413, 0.0532, 0.0583, 0.0562, 0.0422, 0.0366, 0.0514, 0.0332, 0.0542, 0.0323]
BLSTM_Att_std_asc = [0.0492, 0.046, 0.0663, 0.0582, 0.0533, 0.042, 0.0572, 0.0299, 0.0536, 0.0491]
GRU_std_asc = [0.0492, 0.0447, 0.0493, 0.0454, 0.0343, 0.0373, 0.0541, 0.0362, 0.0578, 0.0553]
GRU_Att_std_asc = [0.0502, 0.0516, 0.0546, 0.0532, 0.0535, 0.0276, 0.0464, 0.032, 0.0534, 0.0503]
BGRU_std_asc = [0.0517, 0.069, 0.0532, 0.0512, 0.0341, 0.0515, 0.0498, 0.0325, 0.0595, 0.052]
BGRU_Att_std_asc = [0.0576, 0.0403, 0.0488, 0.0647, 0.0446, 0.0403, 0.0499, 0.0397, 0.0487, 0.0545]

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
    plt.figure()
    # 设置x轴坐标名称
    names = [""]  # x轴坐标名称
    names = names + all_name_list[i]
    name_range = range(len(names))  # x轴坐标名称对于的range索引
    x = np.array([i for i in range(1, 11)])  # x轴坐标画图位置
    y = np.array(all_TSS_list[i])  # 个人TSS分数
    x_err = np.array(all_std_list[i])  # 个人TSS分数标准差

    # 开始画图
    plt.title(all_title_list[i]+'≥M')
    plt.ylim((0, 0.8))
    plt.xlim((0, 11))
    plt.xticks(name_range, names, rotation=90)  #
    plt.bar(x, y, yerr=x_err, ecolor='black')
    plt.tight_layout()  # 调整画布大小，防止显示不全
    plt.savefig('D:/m_only/'+str(i)+'_M_only_' + all_title_list[i] + '.jpg')
    plt.show()

# 调整子图形之间的纵向距离
# figure.subplots_adjust(hspace=0.1)

# plt.errorbar(x, y, yerr=x_err, ecolor='black', color='red', linestyle='--')
