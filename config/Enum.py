class FeatureEnum(object):
    def __init__(self):
        self.TOTUSJH = 'TOTUSJH'
        self.TOTPOT = 'TOTPOT'
        self.TOTUSJZ = 'TOTUSJZ'
        self.ABSNJZH = 'ABSNJZH'
        self.SAVNCPP = 'SAVNCPP'
        self.USFLUX = 'USFLUX'
        self.AREA_ACR = 'AREA_ACR'
        self.MEANPOT = 'MEANPOT'
        self.R_VALUE = 'R_VALUE'
        self.SHRGT45 = 'SHRGT45'


class ModelType(object):
    def __init__(self):
        self.NN = "NN"
        # 下面是LSTM模型名称
        self.LSTM = "LSTM"
        self.LSTM_attention = "LSTM_attention"
        self.LSTM_attention2 = "LSTM_attention2"
        self.Bi_LSTM = "Bi_LSTM"
        self.Bi_LSTM_attention = "Bi_LSTM_attention"
        self.Bi_LSTM_attention2 = "Bi_LSTM_attention2"
        # 下面是GRU模型名称
        self.GRU = "GRU"
        self.GRU_attention = "GRU_attention"
        self.GRU_attention2 = "GRU_attention2"
        self.Bi_GRU = "Bi_GRU"
        self.Bi_GRU_attention = "Bi_GRU_attention"
        self.Bi_GRU_attention2 = "Bi_GRU_attention2"
