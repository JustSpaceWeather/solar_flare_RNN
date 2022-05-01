# common
def get_model_path(p, data_type, class_type, model_name, time_steps, index):
    if model_name == 'NN':
        return p + '/weights/' + data_type + '/NN_best≥' + class_type + '_time_steps=' + str(
            time_steps) + '/NN_' + class_type + '_best_' + str(index) + '.h5'
    else:
        return p + '/weights/' + data_type + '/' + model_name + '_best≥' + class_type + '/time_steps=' + str(
            time_steps) + '/' + model_name + '_' + class_type + '_' + str(time_steps) + '_best_' + str(index) + '.h5'


def get_one_feature_model_path(p, data_type, class_type, model_name, time_steps, index, feature_name):
    path = p + '/weights/feature_impotence/' + data_type
    if model_name == 'NN':
        path = path + '/NN_best≥' + class_type + '_time_steps=' + str(
            time_steps) + '/' + feature_name + '/NN_' + class_type + '_' + feature_name + '_best_' + str(index) + '.h5'
    else:
        path = path + '/' + model_name + '_best≥' + class_type + '/time_steps=' + str(
            time_steps) + '/' + feature_name + '/' + model_name + '_' + class_type + '_' + str(
            time_steps) + '_' + feature_name + '_best_' + str(index) + '.h5'
    return path


# NN
def get_NN_model_path(p, data_type, class_type, index):
    return get_model_path(p, data_type, class_type, 'NN', 1, index)


# LSTM系列
def get_LSTM_model_path(p, data_type, class_type, time_steps, index):
    return get_model_path(p, data_type, class_type, 'LSTM', time_steps, index)


def get_LSTM_Att_model_path(p, data_type, class_type, time_steps, index):
    return get_model_path(p, data_type, class_type, 'LSTM_attention', time_steps, index)


def get_BLSTM_model_path(p, data_type, class_type, time_steps, index):
    return get_model_path(p, data_type, class_type, 'Bi_LSTM', time_steps, index)


def get_BLSTM_Att_model_path(p, data_type, class_type, time_steps, index):
    return get_model_path(p, data_type, class_type, 'Bi_LSTM_attention', time_steps, index)


# GRU系列
def get_GRU_model_path(p, data_type, class_type, time_steps, index):
    return get_model_path(p, data_type, class_type, 'GRU', time_steps, index)


def get_GRU_Att_model_path(p, data_type, class_type, time_steps, index):
    return get_model_path(p, data_type, class_type, 'GRU_attention', time_steps, index)


def get_BGRU_model_path(p, data_type, class_type, time_steps, index):
    return get_model_path(p, data_type, class_type, 'Bi_GRU', time_steps, index)


def get_BGRU_Att_model_path(p, data_type, class_type, time_steps, index):
    return get_model_path(p, data_type, class_type, 'Bi_GRU_attention', time_steps, index)

# model_path = get_model_path(r'D:\workspace\python-workspace\solar_flare_RNN', '2018', 'C', 'LSTM', 1, 2)
# print(model_path)
