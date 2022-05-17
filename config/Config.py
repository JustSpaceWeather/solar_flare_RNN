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
        self.epoch = 50
        self.batch_size = 64
        self.time_steps_list_120 = [1, 10, 12, 15, 20, 24, 30, 40, 60, 120]
        self.time_steps_list_40 = [1, 2, 5, 10, 20, 40]  # 1, 2, 5, 10, 20, 40
        self.time_steps_list = self.time_steps_list_40
        self.verbose = 1  # 0不显示结果和进度条，1显示结果和进度条，2只显示结果不显示进度条


class DetectConfig(object):
    """
    测试参数配置
    """

    def __init__(self):
        self.time_steps_list_40 = [40]  # 1, 2, 5, 10, 20, 40
        self.time_steps_list_120 = [1, 10, 12, 15, 20, 24, 30, 40, 60, 120]
        self.time_steps_list = self.time_steps_list_40
        self.score_metrics = []


class File202240Config(object):
    """
    20200412_2022降采样到40的train_valid_test文件位置
    """

    def __init__(self, p):
        self.train_file = p + '/data/20220418_202240/train'
        # 参与训练的验证集
        self.valid_file = p + '/data/20220418_202240/valid'
        # self.valid_file = p + '/data/20220418_202240/test'
        # 不参与训练的测试集
        self.test_file = p + '/data/20220418_202240/test'
        # self.test_file = p + '/data/20220418_202240/valid'
