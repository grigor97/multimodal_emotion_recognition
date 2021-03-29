import matplotlib.pyplot as plt


def nn_save_model_plots(model_history, save_path):
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(save_path + "/loss.png")

    plt.clf()

    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']

    plt.plot(epochs, acc, 'ro', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(save_path + "/accuracy.png")
