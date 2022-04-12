import sys
import os

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(p)

from model.Bi_GRU_model import get_Bi_GRU_model
from util.load_data import load_data_list
from util.load_data import load_data_M
from util.load_data import Rectify
from util.set_seed import set_seed
from util.score import show_score_and_save_weights
from util.show_pic_util import save_loss
from config.Config import TrainConfig
from config.Config import TTFileConfig

# 训练参数
file_config = TTFileConfig(p)
config = TrainConfig()
set_seed()

train_list = load_data_list(file_config.train_file)
test_list = load_data_list(file_config.test_file)

best_TSS_dir = {}

if __name__ == '__main__':
    for time_steps in config.time_steps_list:
        is_new = True
        model_save_path = p + '/weights/TT/Bi_GRU_best≥C/time_steps=' + str(time_steps)
        best_TSS_list = []  # 保存每个训练集的最好的TSS
        for i in range(10):
            (x_train, y_train, train_weight_dir), (x_test, y_test, test_weight_dir) = load_data_M(train_list[i],
                                                                                                  test_list[i])
            x_train = x_train.reshape(-1, time_steps, 10)
            x_test = x_test.reshape(-1, time_steps, 10)
            y_train = Rectify(y_train, time_steps)
            y_test = Rectify(y_test, time_steps)
            model = get_Bi_GRU_model(time_steps=time_steps, learning_rate=config.learning_rate,
                                     dropout_rate=config.dropout_rate, seed=config.glorot_normal_seed,
                                     score_metrics=config.score_metrics)
            # 评价指标初始化
            best_TSS = float('-inf')
            loss_list, val_loss_list = [], []
            for j in range(config.epoch):
                print("time_steps =", time_steps)
                print(train_list[i] + '\n' + test_list[i] + '\nEpoch ' + str(j + 1) + '/' + str(
                    config.epoch))  # 打印当前训练的训练集和代数                # 开始训练
                history = model.fit(
                    x_train, y_train,
                    batch_size=config.batch_size,
                    epochs=1,
                    verbose=config.verbose,
                    class_weight=train_weight_dir,  # {dict, 'balanced'},
                    validation_data=(x_test, y_test),
                )
                # 开始评价
                y_true = y_test.argmax(axis=1)  # 真实的标签
                y_pred = model.predict(x_test, batch_size=config.batch_size).argmax(axis=1)  # 将数据传入，得到预测的标签
                best_TSS = show_score_and_save_weights(  # 计算最好的TSS，并保存取得最好的TSS的权重
                    model=model,
                    best_TSS=best_TSS,
                    y_true=y_true, y_pred=y_pred,
                    filename=model_save_path + '/Bi_GRU_M_' + str(time_steps) + '_best_' + str(i) + '.h5'
                )
                loss_list.append(history.history['loss'])
                val_loss_list.append(history.history['val_loss'])
                print('======================================')
            save_loss(  # 保存训练时损失函数的变化图象
                loss_list, val_loss_list, config.epoch,
                model_save_path + '/Bi_GRU_M_' + str(time_steps) + '_best_' + str(i) + '.jpg',
                is_new
            )
            is_new = False
            best_TSS_list.append(best_TSS)
        # 一个时间步全部训练完成后，打印所有最好的TSS
        print('time_steps =', time_steps)
        best_TSS_dir[time_steps] = best_TSS_list
        for best_TSS in best_TSS_list:
            print(best_TSS)
    print(best_TSS_dir)
