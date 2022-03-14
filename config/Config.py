class SeedConfig(object):
    def __init__(self):
        self.PYTHONHASHSEED = '0'
        self.random_seed = 147
        self.np_random_seed = 258
        self.tf_random_seed = 110


class TrainConfig(object):
    def __init__(self):
        self.learning_rate = 1e-3
        self.dropout_rate = 0.5
        self.glorot_normal_seed = 369
        self.score_metrics = ['acc']
        self.epoch = 100
        self.batch_size = 120
        self.time_steps_list = [120]  # 1, 10, 12, 15, 20, 24, 30, 40, 60, 120
        # 0不显示结果和进度条，1显示结果和进度条，2只显示结果不显示进度条
        self.verbose = 1


class TTFileConfig(object):
    """
    train_test文件位置
    """

    def __init__(self, p):
        self.train_file = p + r'\data\TT\30_train_85858585\10_best_train'
        self.test_file = p + r'\data\TT\30_test_85858585\10_best_test'


class TVTFileConfig(object):
    """
    train_valid_test文件位置
    """

    def __init__(self, p):
        self.train_file = ''
        self.valid_file = ''
        self.test_file = ''


class DetectConfig(object):
    def __init__(self):
        self.time_steps_list = [1, 10, 12, 15, 20, 24, 30, 40, 60, 120]  #
        self.score_metrics = []
