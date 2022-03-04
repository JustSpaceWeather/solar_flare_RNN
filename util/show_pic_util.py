import matplotlib.pyplot as plt


def show_loss(history, epoch):
    # 迭代图像
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epoch)
    plt.plot(epochs_range, loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Test Loss')
    plt.legend(loc='upper right')
    plt.title('Train and Val Loss')
    plt.show()
