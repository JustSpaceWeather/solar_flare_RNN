import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

from config.Config import TrainConfig
from model import NN_model, LSTM_model, Bi_LSTM_model, LSTM_attention_model, Bi_LSTM_attention_model, GRU_model, \
    Bi_GRU_model, GRU_attention_model, Bi_GRU_attention_model


def save_show_model(model, file_name):
    plot_model(
        model=model,
        to_file='D:/model_pic/' + file_name,
        show_shapes=True,
        show_layer_names=False,
        rankdir='TB'  # 'TB' creates a vertical plot; 'LR' creates a horizontal plot.
    )
    plt.figure(figsize=(10, 10))
    img = plt.imread('D:/model_pic/' + file_name)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    config = TrainConfig()
    NN_model = NN_model.get_NN_model(learning_rate=config.learning_rate, dropout_rate=config.dropout_rate,
                                     seed=config.glorot_normal_seed, score_metrics=config.score_metrics)
    LSTM_model = LSTM_model.get_LSTM_model(time_steps=120, learning_rate=config.learning_rate,
                                           dropout_rate=config.dropout_rate, seed=config.glorot_normal_seed,
                                           score_metrics=config.score_metrics)
    GRU_model = GRU_model.get_GRU_model(time_steps=120, learning_rate=config.learning_rate,
                                        dropout_rate=config.dropout_rate, seed=config.glorot_normal_seed,
                                        score_metrics=config.score_metrics)
    Bi_LSTM_model = Bi_LSTM_model.get_Bi_LSTM_model(time_steps=120, learning_rate=config.learning_rate,
                                                    dropout_rate=config.dropout_rate, seed=config.glorot_normal_seed,
                                                    score_metrics=config.score_metrics)
    Bi_GRU_model = Bi_GRU_model.get_Bi_GRU_model(time_steps=120, learning_rate=config.learning_rate,
                                                 dropout_rate=config.dropout_rate, seed=config.glorot_normal_seed,
                                                 score_metrics=config.score_metrics)
    LSTM_attention_model = LSTM_attention_model.get_LSTM_attention_model(time_steps=120,
                                                                         learning_rate=config.learning_rate,
                                                                         dropout_rate=config.dropout_rate,
                                                                         seed=config.glorot_normal_seed,
                                                                         score_metrics=config.score_metrics)
    GRU_attention_model = GRU_attention_model.get_GRU_attention_model(time_steps=120,
                                                                      learning_rate=config.learning_rate,
                                                                      dropout_rate=config.dropout_rate,
                                                                      seed=config.glorot_normal_seed,
                                                                      score_metrics=config.score_metrics)
    Bi_LSTM_attention_model = Bi_LSTM_attention_model.get_Bi_LSTM_attention_model(time_steps=120,
                                                                                  learning_rate=config.learning_rate,
                                                                                  dropout_rate=config.dropout_rate,
                                                                                  seed=config.glorot_normal_seed,
                                                                                  score_metrics=config.score_metrics)
    Bi_GRU_attention_model = Bi_GRU_attention_model.get_Bi_GRU_attention_model(time_steps=120,
                                                                               learning_rate=config.learning_rate,
                                                                               dropout_rate=config.dropout_rate,
                                                                               seed=config.glorot_normal_seed,
                                                                               score_metrics=config.score_metrics)
    save_show_model(NN_model, 'NN_model.png')
    save_show_model(LSTM_model, 'LSTM_model.png')
    save_show_model(GRU_model, 'GRU_model.png')
    save_show_model(Bi_LSTM_model, 'Bi_LSTM_model.png')
    save_show_model(Bi_GRU_model, 'Bi_GRU_model.png')
    save_show_model(LSTM_attention_model, 'LSTM_attention_model.png')
    save_show_model(GRU_attention_model, 'GRU_attention_model.png')
    save_show_model(Bi_LSTM_attention_model, 'Bi_LSTM_attention_model.png')
    save_show_model(Bi_GRU_attention_model, 'Bi_GRU_attention_model.png')
