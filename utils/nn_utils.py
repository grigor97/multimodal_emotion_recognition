import numpy as np
from utils.utils import *
import matplotlib.pyplot as plt


def nn_save_model_plots(model_history, save_path):
    """
    Saves neural network loss and accuracy plots
    :param model_history: trained model history
    :param save_path: path to save plots
    :return: None
    """
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


def normalize_data(audio_train, audio_test):
    """
    Normalizes dataset with mean zero and variance 1
    :param audio_train: train data for audio
    :param audio_test: test data for audio
    :return: normalized data
    """
    train_mean = audio_train.mean(axis=0)
    audio_train -= train_mean
    audio_test -= train_mean
    audio_train / audio_train.sum(axis=1).reshape((audio_train.shape[0], 1))
    audio_test / audio_test.sum(axis=1).reshape((audio_test.shape[0], 1))

    return audio_train, audio_test


def load_video_data(config):
    """
    Loads train and test data for video models
    :param config: configuration file
    :return: train and test data
    """
    train_pkl = config['data']['train_pkl']
    test_pkl = config['data']['test_pkl']

    train = load_pickle(train_pkl)
    test = load_pickle(test_pkl)
    # loading datasets
    audio_train = train['train_audio_data']
    # audio_train = np.array(audio_train)
    pic_train = train['train_pic_data']
    labels_train = train['train_label_data']

    audio_test = test['test_audio_data']
    # audio_test = np.array(audio_test)
    pic_test = test['test_pic_data']
    labels_test = test['test_label_data']

    audio_train, audio_test = normalize_data(audio_train, audio_test)

    print("shapes of train is {}, {} and shape of label is {}".format(audio_train.shape,
                                                                      pic_train.shape,
                                                                      labels_train.shape))
    print("shapes of test is {}, {} and shape of label is {}".format(audio_test.shape,
                                                                     pic_test.shape,
                                                                     labels_test.shape))

    train_data = (audio_train, pic_train, labels_train)
    test_data = (audio_test, pic_test, labels_test)

    return train_data, test_data


def load_audio_data(config):
    """
    Loads train and test data for audio models
    :param config: configuration file
    :return: train and test data
    """
    train_pkl = config['data']['train_pkl']
    test_pkl = config['data']['test_pkl']

    train = load_pickle(train_pkl)
    test = load_pickle(test_pkl)

    # loading datasets
    audio_train = train['train_audio_data']
    # audio_train = np.array(audio_train)
    labels_train = train['train_label_data']

    audio_test = test['test_audio_data']
    # audio_test = np.array(audio_test)
    labels_test = test['test_label_data']

    audio_train, audio_test = normalize_data(audio_train, audio_test)

    print("shapes of train is {} and shape of label is {}".format(audio_train.shape,
                                                                  labels_train.shape))
    print("shapes of test is {} and shape of label is {}".format(audio_test.shape,
                                                                 labels_test.shape))

    train_data = (np.expand_dims(audio_train, -1), labels_train)
    test_data = (np.expand_dims(audio_test, -1), labels_test)

    return train_data, test_data
