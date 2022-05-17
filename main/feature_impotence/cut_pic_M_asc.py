from config.Enum import *
import matplotlib.pyplot as plt
import numpy as np

"""
蓝色条代表平均个人TSS分数，红色折线代表特征的平均累积TSS分数。
"""
figure_cut_M_asc = feature_M_asc
figure_cut_M_desc = feature_M_desc

all_title_list = [
    'NN', 'LSTM', 'LSTM_Att', 'BLSTM', 'BLSTM_Att', 'GRU', 'GRU_Att', 'BGRU', 'BGRU_Att'
]

M_TSS_asc_NN = [0.67,0.679285714,0.682857143,0.677857143,0.675714286,0.57,0.57,0.469285714,0.391428571,0.294285714
]
M_TSS_asc_LSTM = [0.757142857,0.757857143,0.757142857,0.7,0.674285714,0.65,0.645714286,0.569285714,0.502857143,0.322857143
]
M_TSS_asc_LSTM_Att = [0.746428571,0.751428571,0.752142857,0.692142857,0.671428571,0.647142857,0.650714286,0.58,0.485,0.342857143
]
M_TSS_asc_BLSTM = [0.761428571,0.75,0.765,0.699285714,0.698571429,0.649285714,0.639285714,0.566428571,0.503571429,0.343571429
]
M_TSS_asc_BLSTM_Att = [0.758571429,0.754285714,0.752142857,0.697857143,0.667142857,0.645714286,0.641428571,0.565714286,0.476428571,0.332857143
]
M_TSS_asc_GRU = [0.757142857,0.749285714,0.742142857,0.71,0.675,0.646428571,0.647857143,0.567142857,0.504285714,0.321428571
]
M_TSS_asc_GRU_Att = [0.745,0.74,0.744285714,0.706428571,0.675,0.648571429,0.652857143,0.565,0.486428571,0.331428571
]
M_TSS_asc_BGRU = [0.759285714,0.75,0.742857143,0.702142857,0.685,0.664285714,0.655,0.554285714,0.465714286,0.337142857
]
M_TSS_asc_BGRU_Att = [0.758571429,0.751428571,0.759285714,0.692142857,0.686428571,0.647142857,0.642142857,0.578571429,0.495714286,0.322142857
]

M_std_asc_NN = [0.049877401,0.08369344,0.053946307,0.054720999,0.068452277,0.062662035,0.0678233,0.067616113,0.071127939,0.055402056
]
M_std_asc_LSTM = [0.024117061,0.039441251,0.03683942,0.057232073,0.054229294,0.066008657,0.064618188,0.05200569,0.069414461,0.051090156
]
M_std_asc_LSTM_Att = [0.043005695,0.043659162,0.045518554,0.061233078,0.059074495,0.06663945,0.075690697,0.072478005,0.051155026,0.044031529
]
M_std_asc_BLSTM = [0.025793529,0.025152596,0.037722131,0.052435578,0.069897885,0.064321419,0.066182339,0.069696866,0.064067089,0.055185217
]
M_std_asc_BLSTM_Att = [0.031102202,0.038756171,0.044840989,0.061449332,0.061957853,0.059178043,0.060423674,0.051587057,0.068067344,0.039795395
]
M_std_asc_GRU = [0.03380617,0.037314165,0.037991675,0.052508503,0.068157231,0.066796218,0.073889741,0.052411247,0.057870369,0.049692935
]
M_std_asc_GRU_Att = [0.025364687,0.046751623,0.046379095,0.05628372,0.060714286,0.069311483,0.067627431,0.061813578,0.048175911,0.043236417
]
M_std_asc_BGRU = [0.043688368,0.048550416,0.032888184,0.052201533,0.07308438,0.072562429,0.065391287,0.048149427,0.07040698,0.047466422
]
M_std_asc_BGRU_Att = [0.033166248,0.042833327,0.035290341,0.047750414,0.060394114,0.069854075,0.065344456,0.056694671,0.042039826,0.042552489
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

M_TSS_All_asc = [
    M_TSS_asc_NN,
    M_TSS_asc_LSTM,
    M_TSS_asc_LSTM_Att,
    M_TSS_asc_BLSTM,
    M_TSS_asc_BLSTM_Att,
    M_TSS_asc_GRU,
    M_TSS_asc_GRU_Att,
    M_TSS_asc_BGRU,
    M_TSS_asc_BGRU_Att,
]

M_std_All_asc = [
    M_std_asc_NN,
    M_std_asc_LSTM,
    M_std_asc_LSTM_Att,
    M_std_asc_BLSTM,
    M_std_asc_BLSTM_Att,
    M_std_asc_GRU,
    M_std_asc_GRU_Att,
    M_std_asc_BGRU,
    M_std_asc_BGRU_Att,
]

for i in range(len(M_std_All_asc)):
    plt.figure(dpi=400)
    # 设置x轴坐标名称
    names = [""]  # x轴坐标名称
    names = names + all_name_list[i]
    name_range = range(len(names))  # x轴坐标名称对于的range索引
    x = np.array([i for i in range(1, 11)])  # x轴坐标画图位置
    y = np.array(M_TSS_All_asc[i])  # 个人TSS分数
    x_err = np.array(M_std_All_asc[i])  # 个人TSS分数标准差

    # 开始画图
    plt.title(all_title_list[i]+'≥M')
    plt.ylim((0.2, 0.85))
    plt.xlim((0, 11))
    plt.xticks(name_range, names, rotation=90)  #
    # plt.bar(x, y, yerr=x_err, ecolor='black')
    plt.errorbar(x, y, yerr=x_err, ecolor='black', color='red', linestyle='--')
    plt.tight_layout()  # 调整画布大小，防止显示不全
    plt.savefig('D:/M_cut_asc/'+str(i)+'_M_cut_asc_' + all_title_list[i] + '.jpg')
    plt.show()

# 调整子图形之间的纵向距离
# figure.subplots_adjust(hspace=0.1)
