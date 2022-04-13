from model.Bi_LSTM_attention_model import get_Bi_LSTM_attention_model
from util.load_data import load_data_list
from util.load_data import load_data_C
from util.load_data import Rectify
from util.set_seed import set_seed
from util.score import show_score_and_save_weights
from util.show_pic_util import save_loss
from config.Config import TrainConfig

import keras.backend as K


def Bi_LSTM_attention_C_train(p: str, file_config, train_type: str):
    """
    :param p: 根目录地址
    :param file_config: train_config.Config
    :param train_type: TT, TVT, 2018, 2020
    :return:
    """
    pass
    # 训练参数
    train_config = TrainConfig()
    set_seed()

    train_list = load_data_list(file_config.train_file)
    valid_list = load_data_list(file_config.valid_file)

    best_TSS_dir = {}
    for time_steps in train_config.time_steps_list:
        is_new = True
        best_TSS_list = []  # 保存每个训练集的最好的TSS
        model_save_path = p + '/weights/' + train_type + '/Bi_LSTM_attention_best≥C/time_steps=' + str(time_steps)
        for i in range(len(train_list)):
            (x_train, y_train, train_weight_dir), (x_valid, y_valid, valid_weight_dir) = load_data_C(train_list[i],
                                                                                                     valid_list[i])
            x_train = x_train.reshape(-1, time_steps, 10)
            x_valid = x_valid.reshape(-1, time_steps, 10)
            y_train = Rectify(y_train, time_steps)
            y_valid = Rectify(y_valid, time_steps)
            model = get_Bi_LSTM_attention_model(
                time_steps=time_steps,
                learning_rate=train_config.learning_rate,
                dropout_rate=train_config.dropout_rate,
                seed=train_config.glorot_normal_seed,
                score_metrics=train_config.score_metrics
            )
            # 评价指标初始化
            best_TSS = float('-inf')
            loss_list, val_loss_list = [], []
            for j in range(train_config.epoch):
                # 打印当前训练的时间步、训练集和代数
                print("time_steps =", time_steps)
                print(train_list[i])
                print(valid_list[i])
                print('Epoch ' + str(j + 1) + '/' + str(train_config.epoch))
                # 开始训练
                history = model.fit(
                    x_train, y_train,
                    batch_size=train_config.batch_size,
                    epochs=1,
                    verbose=train_config.verbose,
                    class_weight=train_weight_dir,
                    validation_data=(x_valid, y_valid),
                )
                # 开始评价
                y_true = y_valid.argmax(axis=1)  # 真实的标签
                y_pred = model.predict(x_valid, batch_size=train_config.batch_size).argmax(axis=1)  # 将数据传入，得到预测的标签
                best_TSS = show_score_and_save_weights(  # 计算最好的TSS，并保存取得最好的TSS的权重
                    model=model,
                    best_TSS=best_TSS,
                    y_true=y_true, y_pred=y_pred,
                    filename=model_save_path + '/Bi_LSTM_attention_C_' + str(time_steps) + '_best_' + str(i) + '.h5'
                )
                loss_list.append(history.history['loss'])
                val_loss_list.append(history.history['val_loss'])
                print('======================================')
            save_loss(
                loss_list, val_loss_list, train_config.epoch,
                model_save_path + '/Bi_LSTM_attention_C_' + str(time_steps) + '_best_' + str(i) + '.jpg',
                is_new
            )
            is_new = False
            best_TSS_list.append(best_TSS)
            K.clear_session()
        # 全部训练完成后，打印所有权重的指标
        print('time_steps =', time_steps)
        best_TSS_dir[time_steps] = best_TSS_list
        for best_TSS in best_TSS_list:
            print(best_TSS)
    print(best_TSS_dir)
