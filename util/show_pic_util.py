import matplotlib.pyplot as plt


def show_loss(loss_list, val_loss_list, epoch, file_path):
    plt.figure()
    epochs_range = range(epoch)
    plt.plot(epochs_range, loss_list, label='Train Loss')
    plt.plot(epochs_range, val_loss_list, label='Valid Loss')
    plt.legend(loc='upper right')
    plt.title('Train and Val Loss')
    plt.savefig(file_path)
