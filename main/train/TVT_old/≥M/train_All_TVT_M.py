import os
import sys

p = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(p)
from config.Config import TVTFileConfig
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

file_config = TVTFileConfig(p)

NN_M_train(p, file_config, 'TVT')
LSTM_M_train(p, file_config, 'TVT')
LSTM_attention_M_train(p, file_config, 'TVT')
LSTM_attention2_M_train(p, file_config, 'TVT')
GRU_M_train(p, file_config, 'TVT')
GRU_attention_M_train(p, file_config, 'TVT')
GRU_attention2_M_train(p, file_config, 'TVT')
Bi_LSTM_M_train(p, file_config, 'TVT')
Bi_LSTM_attention_M_train(p, file_config, 'TVT')
Bi_LSTM_attention2_M_train(p, file_config, 'TVT')
Bi_GRU_M_train(p, file_config, 'TVT')
Bi_GRU_attention_M_train(p, file_config, 'TVT')
Bi_GRU_attention2_M_train(p, file_config, 'TVT')
