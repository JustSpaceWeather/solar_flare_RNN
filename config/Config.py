class SeedConfig(object):
    """
    随机数种子配置
    """

    def __init__(self):
        self.PYTHONHASHSEED = '0'
        self.random_seed = 147
        self.np_random_seed = 258
        self.tf_random_seed = 110


class TrainConfig(object):
    """
    训练参数配置
    """

    def __init__(self):
        self.learning_rate = 1e-3
        self.dropout_rate = 0.5
        self.glorot_normal_seed = 369
        self.score_metrics = ['acc']  # , TP, TN
        self.epoch = 1
        self.batch_size = 64
        self.time_steps_list_120 = [1, 10, 12, 15, 20, 24, 30, 40, 60, 120]
        self.time_steps_list_40 = [40]  # 1, 2, 5, 10, 20,
        self.time_steps_list = self.time_steps_list_40
        self.verbose = 1  # 0不显示结果和进度条，1显示结果和进度条，2只显示结果不显示进度条


class DetectConfig(object):
    """
    测试参数配置
    """

    def __init__(self):
        self.time_steps_list_40 = [40]  # 1, 2, 5, 10, 20, 40
        self.time_steps_list_120 = [1, 10, 12, 15, 20, 24, 30, 40, 60, 120]
        self.time_steps_list = self.time_steps_list_120
        self.score_metrics = []


# 文件位置配置
class TTFileConfig(object):
    """
    train_test文件位置
    """

    def __init__(self, p):
        self.train_file = p + '/data/20220102_TT/30_train_85858585/10_best_train'
        self.test_file = p + '/data/20220102_TT/30_test_85858585/10_best_test'
        self.valid_file = self.test_file


class TVTFileConfig(object):
    """
    train_valid_test文件位置
    """

    def __init__(self, p):
        self.train_file = p + '/data/20220102_TVT/train'
        self.valid_file = p + '/data/20220102_TVT/valid'
        self.test_file = p + '/data/20220102_TVT/test'


class File2018Config(object):
    """
    20200412_2018train_valid_test文件位置
    """

    def __init__(self, p):
        self.train_file = p + '/data/20200412_2018/train'
        self.valid_file = p + '/data/20200412_2018/valid'
        self.test_file = p + '/data/20200412_2018/test'


class File2022Config(object):
    """
    20200412_2022train_valid_test文件位置
    """

    def __init__(self, p):
        self.train_file = p + '/data/20200412_2022/train'
        self.valid_file = p + '/data/20200412_2022/valid'
        self.test_file = p + '/data/20200412_2022/test'


class File202240Config(object):
    """
    20200412_2022降采样到40的train_valid_test文件位置
    """

    def __init__(self, p):
        self.train_file = p + '/data/20220418_202240/train'
        self.valid_file = p + '/data/20220418_202240/valid'
        # self.test_file = p + '/data/20220418_202240/test'
        self.test_file = p + '/data/20220418_202240/valid'


class ModelType(object):
    def __init__(self):
        self.NN = "NN"
        # 下面是LSTM模型名称
        self.LSTM = "LSTM"
        self.Bi_LSTM = "Bi_LSTM"
        self.LSTM_attention = "LSTM_attention"
        self.LSTM_attention2 = "LSTM_attention2"
        self.Bi_LSTM_attention = "Bi_LSTM_attention"
        self.Bi_LSTM_attention2 = "Bi_LSTM_attention2"
        # 下面是GRU模型名称
        self.GRU = "GRU"
        self.Bi_GRU = "Bi_GRU"
        self.GRU_attention = "GRU_attention"
        self.GRU_attention2 = "GRU_attention2"
        self.Bi_GRU_attention = "Bi_GRU_attention"
        self.Bi_GRU_attention2 = "Bi_GRU_attention2"
