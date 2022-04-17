import os
import sys

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(p)

from util.load_data import load_data_M
from model.LSTM_model import get_LSTM_model
from config.Config import TrainConfig
from util.score import show_score_and_save_weights
from util.load_data import Rectify

if __name__ == '__main__':
    for n in range(1, 11, 1):
        train_config = TrainConfig()
        model = get_LSTM_model(time_steps=120, learning_rate=train_config.learning_rate,
                               dropout_rate=train_config.dropout_rate, seed=train_config.glorot_normal_seed,
                               score_metrics=train_config.score_metrics)
        for i in range(1, 2, 1):
            train_path = 'D:/' + str(n) + '/train_'
            valid_path = 'D:/' + str(n) + '/valid_'
            train_path = train_path + str(i) + '.csv'
            valid_path = valid_path + str(i) + '.csv'
            (x_train, y_train, train_weight), (x_valid, y_valid, valid_weight) = load_data_M(train_path, valid_path)
            x_train = x_train.reshape(-1, 120, 10)
            x_valid = x_valid.reshape(-1, 120, 10)
            y_train = Rectify(y_train, 120)
            y_valid = Rectify(y_valid, 120)
            # 评价指标初始化
            best_TSS = float('-inf')
            for j in range(100):
                print("time_steps =", 120)
                print(train_path + '\n' + valid_path + '\nEpoch ' + str(j + 1) + '/' + str(100))  # 打印当前训练的训练集和代数
                model.fit(
                    x_train, y_train,
                    batch_size=120,
                    epochs=1,
                    verbose=1,
                    class_weight=train_weight,
                    validation_data=(x_valid, y_valid),
                )
                # 开始评价
                y_true = y_valid.argmax(axis=1)  # 真实的标签
                y_pred = model.predict(x_valid).argmax(axis=1)  # 将数据传入，得到预测的标签
                best_TSS = show_score_and_save_weights(  # 计算最好的TSS，并保存取得最好的TSS的权重
                    model=model,
                    best_TSS=best_TSS,
                    y_true=y_true, y_pred=y_pred,
                    filename='D:/' + str(n) + '/M' + str(i) + '.h5'
                )
                print('======================================')
