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

        self.all_feature_list = [self.TOTUSJH, self.TOTPOT, self.TOTUSJZ, self.ABSNJZH, self.SAVNCPP,
                                 self.USFLUX, self.AREA_ACR, self.MEANPOT, self.R_VALUE, self.SHRGT45]
        # self.TOTUSJH, self.TOTPOT, self.TOTUSJZ, self.ABSNJZH, self.SAVNCPP,self.USFLUX,
        # self.AREA_ACR, self.MEANPOT, self.R_VALUE, self.SHRGT45


class ModelType(object):
    def __init__(self):
        self.NN = "NN"
        # 下面是LSTM系列模型名称
        self.LSTM = "LSTM"
        self.LSTM_attention = "LSTM_attention"
        self.LSTM_attention2 = "LSTM_attention2"
        self.Bi_LSTM = "Bi_LSTM"
        self.Bi_LSTM_attention = "Bi_LSTM_attention"
        self.Bi_LSTM_attention2 = "Bi_LSTM_attention2"
        # 下面是GRU系列模型名称
        self.GRU = "GRU"
        self.GRU_attention = "GRU_attention"
        self.GRU_attention2 = "GRU_attention2"
        self.Bi_GRU = "Bi_GRU"
        self.Bi_GRU_attention = "Bi_GRU_attention"
        self.Bi_GRU_attention2 = "Bi_GRU_attention2"


feature_C_asc = {  # 从最好到最差
    'NN': ['R_VALUE', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTPOT', 'SAVNCPP', 'AREA_ACR', 'ABSNJZH', 'SHRGT45', 'MEANPOT'],
    'LSTM': ['R_VALUE', 'USFLUX', 'TOTUSJH', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
             'SHRGT45'],
    'GRU': ['R_VALUE', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
            'SHRGT45'],
    'BLSTM': ['R_VALUE', 'USFLUX', 'TOTUSJH', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
              'SHRGT45'],
    'BGRU': ['R_VALUE', 'USFLUX', 'TOTUSJH', 'TOTUSJZ', 'SAVNCPP', 'ABSNJZH', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
             'SHRGT45'],
    'LSTM_Att': ['R_VALUE', 'USFLUX', 'TOTUSJH', 'TOTUSJZ', 'SAVNCPP', 'ABSNJZH', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                 'SHRGT45'],
    'GRU_Att': ['R_VALUE', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                'SHRGT45'],
    'BLSTM_Att': ['R_VALUE', 'USFLUX', 'TOTUSJH', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                  'SHRGT45'],
    'BGRU_Att': ['R_VALUE', 'TOTUSJH', 'USFLUX', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                 'SHRGT45'],
    'LSTM_Att2': ['R_VALUE', 'USFLUX', 'TOTUSJH', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                  'SHRGT45'],
    'GRU_Att2': ['R_VALUE', 'TOTUSJH', 'TOTUSJZ', 'ABSNJZH', 'USFLUX', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                 'SHRGT45'],
    'BLSTM_Att2': ['R_VALUE', 'TOTUSJH', 'USFLUX', 'TOTUSJZ', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                   'SHRGT45'],
    'BGRU_Att2': ['R_VALUE', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'ABSNJZH', 'SAVNCPP', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                  'SHRGT45']
}

feature_C_desc = {  # 从最差到最好
    'NN': ['MEANPOT', 'SHRGT45', 'ABSNJZH', 'AREA_ACR', 'SAVNCPP', 'TOTPOT', 'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'R_VALUE'],
    'LSTM': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'SAVNCPP', 'ABSNJZH', 'TOTUSJZ', 'TOTUSJH', 'USFLUX',
             'R_VALUE'],
    'GRU': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'SAVNCPP', 'ABSNJZH', 'USFLUX', 'TOTUSJZ', 'TOTUSJH',
            'R_VALUE'],
    'BLSTM': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'SAVNCPP', 'ABSNJZH', 'TOTUSJZ', 'TOTUSJH', 'USFLUX',
              'R_VALUE'],
    'BGRU': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'ABSNJZH', 'SAVNCPP', 'TOTUSJZ', 'TOTUSJH', 'USFLUX',
             'R_VALUE'],
    'LSTM_Att': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'ABSNJZH', 'SAVNCPP', 'TOTUSJZ', 'TOTUSJH', 'USFLUX',
                 'R_VALUE'],
    'GRU_Att': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'SAVNCPP', 'ABSNJZH', 'USFLUX', 'TOTUSJZ', 'TOTUSJH',
                'R_VALUE'],
    'BLSTM_Att': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'SAVNCPP', 'ABSNJZH', 'TOTUSJZ', 'TOTUSJH', 'USFLUX',
                  'R_VALUE'],
    'BGRU_Att': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'SAVNCPP', 'ABSNJZH', 'TOTUSJZ', 'USFLUX', 'TOTUSJH',
                 'R_VALUE'],
    'LSTM_Att2': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'SAVNCPP', 'ABSNJZH', 'TOTUSJZ', 'TOTUSJH', 'USFLUX',
                  'R_VALUE'],
    'GRU_Att2': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'SAVNCPP', 'USFLUX', 'ABSNJZH', 'TOTUSJZ', 'TOTUSJH',
                 'R_VALUE'],
    'BLSTM_Att2': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'SAVNCPP', 'ABSNJZH', 'TOTUSJZ', 'USFLUX', 'TOTUSJH',
                   'R_VALUE'],
    'BGRU_Att2': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'SAVNCPP', 'ABSNJZH', 'USFLUX', 'TOTUSJZ', 'TOTUSJH',
                  'R_VALUE']
}

feature_M_asc = {  # 从最差到最好
    'NN': ['R_VALUE', 'ABSNJZH', 'TOTUSJH', 'SAVNCPP', 'TOTUSJZ', 'TOTPOT', 'USFLUX', 'AREA_ACR', 'MEANPOT', 'SHRGT45'],
    'LSTM': ['R_VALUE', 'SAVNCPP', 'ABSNJZH', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
             'SHRGT45'],
    'GRU': ['R_VALUE', 'SAVNCPP', 'ABSNJZH', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
            'SHRGT45'],
    'BLSTM': ['R_VALUE', 'SAVNCPP', 'ABSNJZH', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
              'SHRGT45'],
    'BGRU': ['R_VALUE', 'SAVNCPP', 'ABSNJZH', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
             'SHRGT45'],
    'LSTM_Att': ['R_VALUE', 'SAVNCPP', 'ABSNJZH', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                 'SHRGT45'],
    'GRU_Att': ['R_VALUE', 'SAVNCPP', 'ABSNJZH', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                'SHRGT45'],
    'BLSTM_Att': ['R_VALUE', 'SAVNCPP', 'ABSNJZH', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                  'SHRGT45'],
    'BGRU_Att': ['R_VALUE', 'SAVNCPP', 'ABSNJZH', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                 'SHRGT45'],
    'LSTM_Att2': ['R_VALUE', 'SAVNCPP', 'ABSNJZH', 'TOTUSJH', 'TOTUSJZ', 'TOTPOT', 'USFLUX', 'AREA_ACR', 'MEANPOT',
                  'SHRGT45'],
    'BLSTM_Att2': ['R_VALUE', 'SAVNCPP', 'ABSNJZH', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                   'SHRGT45'],
    'GRU_Att2': ['R_VALUE', 'SAVNCPP', 'ABSNJZH', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                 'SHRGT45'],
    'BGRU_Att2': ['R_VALUE', 'SAVNCPP', 'ABSNJZH', 'TOTUSJH', 'TOTUSJZ', 'USFLUX', 'TOTPOT', 'AREA_ACR', 'MEANPOT',
                  'SHRGT45']
}

feature_M_desc = {  # 从最好到最差
    'NN': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'USFLUX', 'TOTPOT', 'TOTUSJZ', 'SAVNCPP', 'TOTUSJH', 'ABSNJZH', 'R_VALUE'],
    'LSTM': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP',
             'R_VALUE'],
    'GRU': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP',
            'R_VALUE'],
    'BLSTM': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP',
              'R_VALUE'],
    'BGRU': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP',
             'R_VALUE'],
    'LSTM_Att': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP',
                 'R_VALUE'],
    'GRU_Att': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP',
                'R_VALUE'],
    'BLSTM_Att': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP',
                  'R_VALUE'],
    'BGRU_Att': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP',
                 'R_VALUE'],
    'LSTM_Att2': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'USFLUX', 'TOTPOT', 'TOTUSJZ', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP',
                  'R_VALUE'],
    'BLSTM_Att2': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP',
                   'R_VALUE'],
    'GRU_Att2': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP',
                 'R_VALUE'],
    'BGRU_Att2': ['SHRGT45', 'MEANPOT', 'AREA_ACR', 'TOTPOT', 'USFLUX', 'TOTUSJZ', 'TOTUSJH', 'ABSNJZH', 'SAVNCPP',
                  'R_VALUE'],
}
