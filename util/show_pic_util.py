import matplotlib.pyplot as plt


def save_loss(loss_list, val_loss_list, epoch, file_path, new_figure: bool):
    if new_figure:
        plt.figure()
    epochs_range = range(epoch)
    plt.plot(epochs_range, loss_list)
    plt.plot(epochs_range, val_loss_list)

    # plt.legend(loc='upper right')
    plt.title('Train and Val Loss')
    plt.savefig(file_path)

