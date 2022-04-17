import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)
from config.Config import TTFileConfig
from common.train_common.M.NN_train_M import NN_M_train
from common.train_common.M.LSTM_train_M import LSTM_M_train
from common.train_common.M.LSTM_attention_train_M import LSTM_attention_M_train
from common.train_common.M.LSTM_attention2_train_M import LSTM_attention2_M_train
from common.train_common.M.GRU_attention_train_M import GRU_attention_M_train
from common.train_common.M.GRU_train_M import GRU_M_train
from common.train_common.M.GRU_attention2_train_M import GRU_attention2_M_train
from common.train_common.M.Bi_LSTM_train_M import Bi_LSTM_M_train
from common.train_common.M.Bi_LSTM_attention_train_M import Bi_LSTM_attention_M_train
from common.train_common.M.Bi_LSTM_attention2_train_M import Bi_LSTM_attention2_M_train
from common.train_common.M.Bi_GRU_train_M import Bi_GRU_M_train
from common.train_common.M.Bi_GRU_attention_train_M import Bi_GRU_attention_M_train
from common.train_common.M.Bi_GRU_attention2_train_M import Bi_GRU_attention2_M_train

file_config = TTFileConfig(p)

NN_M_train(p, file_config, 'TT')
LSTM_M_train(p, file_config, 'TT')
LSTM_attention_M_train(p, file_config, 'TT')
LSTM_attention2_M_train(p, file_config, 'TT')
GRU_M_train(p, file_config, 'TT')
GRU_attention_M_train(p, file_config, 'TT')
GRU_attention2_M_train(p, file_config, 'TT')
Bi_LSTM_M_train(p, file_config, 'TT')
Bi_LSTM_attention_M_train(p, file_config, 'TT')
Bi_LSTM_attention2_M_train(p, file_config, 'TT')
Bi_GRU_M_train(p, file_config, 'TT')
Bi_GRU_attention_M_train(p, file_config, 'TT')
Bi_GRU_attention2_M_train(p, file_config, 'TT')
