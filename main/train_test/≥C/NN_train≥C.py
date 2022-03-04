import sys
import os

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(p)

from model.NN_model import get_NN_model
from util.load_data import load_data_list
from util.load_data import load_data_C
from util.set_seed import set_seed
from util.score import show_score_and_save_weights
from config.Config import TrainConfig

set_seed()

train_list = load_data_list(p + r'\data\30_train_85858585\10_best_train')
test_list = load_data_list(p + r'\data\30_test_85858585\10_best_test')

# 训练参数
config = TrainConfig()
time_steps = 1
best_TSS_list = []  # 保存每个训练集的最好的TSS

if __name__ == '__main__':
    for i in range(10):
        (x_train, y_train, train_weight_dir), (x_test, y_test, test_weight_dir) = load_data_C(train_list[i],
                                                                                              test_list[i])
        model = get_NN_model(
            learning_rate=config.learning_rate,
            dropout_rate=config.dropout_rate,
            glorot_normal_seed=config.glorot_normal_seed,
            score_metrics=config.score_metrics
        )
        # 评价指标初始化
        best_TSS = float('-inf')
        for j in range(config.epoch):
            print(train_list[i] + '\nEpoch ' + str(j) + '/' + str(config.epoch))  # 打印当前训练的训练集和代数
            # 开始训练
            model.fit(
                x_train, y_train,
                batch_size=config.batch_size,
                epochs=1,
                verbose=1,
                class_weight=train_weight_dir,  # {dict, 'balanced'},
                validation_data=(x_test, y_test),
            )
            # 开始评价
            y_true = y_test.argmax(axis=1)  # 真实的标签
            y_pred = model.predict(x_test, batch_size=config.batch_size).argmax(axis=1)  # 将数据传入，得到预测的标签
            each_model_save_path = p + '/weights/NN_best≥C_time_steps=' + str(time_steps)
            best_TSS = show_score_and_save_weights(  # 计算最好的TSS，并保存取得最好的TSS的权重
                model=model,
                best_TSS=best_TSS,
                y_true=y_true, y_pred=y_pred,
                filename=each_model_save_path + '/NN_C_best_' + str(i) + '.h5'
            )
            print('======================================')
        best_TSS_list.append(best_TSS)
    # 全部训练完成后，打印所有权重的指标
    for best_TSS in best_TSS_list:
        print(best_TSS)
