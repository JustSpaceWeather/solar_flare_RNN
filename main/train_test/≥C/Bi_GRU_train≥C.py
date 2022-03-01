import sys
import os

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(p)

from model.Bi_GRU_model import get_Bi_GRU_model
from util.load_data import load_data_list
from util.load_data import load_data_C
from util.load_data import Rectify
from util.set_seed import set_seed
from util.score import show_score_and_save_weights

# 训练参数
learning_rate = 1e-3
dropout_rate = 0.5
glorot_normal_seed = 369
score_metrics = []
epoch = 100
batch_size = 120
time_steps_list = [1, 10, 12, 15, 20, 24, 30, 40, 60, 120]
set_seed()

train_list = load_data_list(p + r'\data\30_train_85858585\10_best_train')
test_list = load_data_list(p + r'\data\30_test_85858585\10_best_test')

best_TSS_dir = {}

if __name__ == '__main__':
    for time_steps in time_steps_list:
        best_TSS_list = []  # 保存每个训练集的最好的TSS
        for i in range(10):
            (x_train, y_train, train_weight_dir), (x_test, y_test, test_weight_dir) = load_data_C(train_list[i],
                                                                                                  test_list[i])
            x_train = x_train.reshape(-1, time_steps, 10)
            x_test = x_test.reshape(-1, time_steps, 10)
            y_train = Rectify(y_train, time_steps)
            y_test = Rectify(y_test, time_steps)
            model = get_Bi_GRU_model(
                time_steps=time_steps,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                glorot_normal_seed=glorot_normal_seed,
                score_metrics=score_metrics
            )
            # 评价指标初始化
            best_TSS = float('-inf')
            for j in range(epoch):
                print("time_steps =", time_steps)
                print(train_list[i] + '\nEpoch ' + str(j) + '/' + str(epoch))  # 打印当前训练的训练集和代数
                # 开始训练
                model.fit(
                    x_train, y_train,
                    batch_size=batch_size,
                    epochs=1,
                    verbose=1,
                    class_weight=train_weight_dir,  # {dict, 'balanced'},
                    validation_data=(x_test, y_test),
                )
                # 开始评价
                y_true = y_test.argmax(axis=1)  # 真实的标签
                y_pred = model.predict(x_test).argmax(axis=1)  # 将数据传入，得到预测的标签
                each_model_save_path = p + '/weights/Bi_GRU_best≥C/time_steps=' + str(time_steps)
                best_TSS = show_score_and_save_weights(  # 计算最好的TSS，并保存取得最好的TSS的权重
                    model=model,
                    best_TSS=best_TSS,
                    y_true=y_true, y_pred=y_pred,
                    filename=each_model_save_path + '/Bi_GRU_C_' + str(time_steps) + '_best_' + str(i) + '.h5'
                )
                print('======================================')
            best_TSS_list.append(best_TSS)
        # 一个时间步全部训练完成后，打印所有最好的TSS
        print('time_steps =', time_steps)
        best_TSS_dir[time_steps] = best_TSS_list
        for best_TSS in best_TSS_list:
            print(best_TSS)
    print(best_TSS_dir)
