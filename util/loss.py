from util.score import BSS


def BSS_loss(y_true, y_pred):
    return 1 - BSS(y_true, y_pred)
