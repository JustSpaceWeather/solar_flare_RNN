from common.detect_common.detect import detect
from config.Config import TrainConfig
from model.NN_model import get_NN_model


def get_and_load_model(time_steps, model_path):
    train_config = TrainConfig()
    model = get_NN_model(
        learning_rate=train_config.learning_rate,
        dropout_rate=0.0,
        seed=train_config.glorot_normal_seed,
        score_metrics=train_config.score_metrics
    )
    model.load_weights(model_path)
    return model


def NN(p, file_config, detect_type, class_type: str) -> None:
    """
    :param p: 项目根目录地址
    :param file_config: 训练文件配置类对象
    :param detect_type: 数据集类型  TT TVT 2018 2022
    :param class_type: 分类类型  C  M
    """
    detect(p, file_config, detect_type, class_type, 'NN', get_and_load_model)
