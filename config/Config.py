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
        self.time_steps_list = [40, 60, 120]  # 1, 10, 12, 15, 20, 24,30 , 40, 60, 120
        self.verbose = 1
