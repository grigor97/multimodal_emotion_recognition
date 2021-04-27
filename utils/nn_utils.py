import numpy as np
import random
from utils.utils import *
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, test_data, tol=0.15, patience=4):
        super(CustomEarlyStopping, self).__init__()
        self.tol = tol
        self.patience = patience
        # self.best_weights = None
        self.test_data = test_data

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        self.te_wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        x, y = self.test_data
        te_loss, te_acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(te_loss, te_acc))

        v_acc = logs.get('val_accuracy')
        t_acc = logs.get('accuracy')

        if v_acc - te_acc > self.tol:
            if v_acc - te_acc > self.tol + 5:
                print('testing gap is too larger wait time is {}'.format(self.te_wait))
                self.te_wait += 100
            print('testing gap is larger wait time is {}'.format(self.te_wait))
            self.te_wait += 1
        else:
            self.te_wait = 0

        if t_acc - v_acc > self.tol:
            if t_acc - v_acc > self.tol + 5:
                print('gap is too larger wait time is {}'.format(self.wait))
                self.wait += 100
            print('gap is larger wait time is {}'.format(self.wait))
            self.wait += 1
        else:
            self.wait = 0

        if (self.wait > self.patience or self.te_wait > self.patience) and epoch > 1:
            print('stopppppppinggggggggg')
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


# class TestCallback(tf.keras.callbacks.Callback):
#     def __init__(self, test_data):
#         self.test_data = test_data
#
#     def on_epoch_end(self, epoch, logs={}):
#         x, y = self.test_data
#         loss, acc = self.model.evaluate(x, y, verbose=0)
#         print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def random_split(audio_x, pic_x, y, spl=0.15):
    """
    Random split of train data into train val
    :param y: labels
    :param audio_x: audio features data
    :param pic_x: picture label data
    :param spl: split percent
    :return: train and validation data
    """
    random.seed(14)
    n = audio_x.shape[0]
    tr_s = int(n * spl)

    pop = range(n)
    val_ind = np.array(random.sample(pop, tr_s))
    train_ind = np.array(list(set(pop).difference(set(val_ind))))

    tr_audio_x = audio_x[train_ind]
    tr_pic_x = pic_x[train_ind]
    tr_y = y[train_ind]
    val_audio_x = audio_x[val_ind]
    val_pic_x = pic_x[val_ind]
    val_y = y[val_ind]

    return tr_audio_x, tr_pic_x, tr_y, val_audio_x, val_pic_x, val_y


def nn_save_model_plots(model_history, save_path):
    """
    Saves neural network loss and accuracy plots
    :param model_history: trained model history
    :param save_path: path to save plots
    :return: None
    """
    print(model_history.history.keys())
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


def normalize_data(audio_train, audio_val, audio_test):
    """
    Normalizes dataset with mean zero and variance 1
    :param audio_val: validation data for audio
    :param audio_train: train data for audio
    :param audio_test: test data for audio
    :return: normalized data
    """
    train_min = audio_train.min(axis=0).reshape((1, audio_train.shape[1]))
    train_denom = audio_train.max(axis=1).reshape((1, audio_train.shape[1])) - audio_train.min(axis=1).reshape((1, audio_train.shape[1]))
    audio_train -= train_min
    audio_val -= train_min
    audio_test -= train_min
    audio_train /= train_denom.reshape((audio_train.shape[0], 1))
    audio_val /= train_denom.reshape((audio_val.shape[0], 1))
    audio_test /= train_denom.reshape((audio_test.shape[0], 1))

    print("after nomilization  ")
    print(train_min)
    print(train_denom)
    print('dsvasva')
    print(audio_train[0, :])
    return audio_train, audio_val, audio_test


def load_data(config):
    """
    Loads train and test data for video models
    :param config: configuration file
    :return: train and test data
    """
    train_pkl = config['data']['train_pkl']
    val_pkl = config['data']['val_pkl']
    test_pkl = config['data']['test_pkl']

    train = load_pickle(train_pkl)
    val = load_pickle(val_pkl)
    test = load_pickle(test_pkl)
    # loading datasets
    audio_train = train['train_audio_data']
    pic_train = train['train_pic_data']
    labels_train = train['train_label_data']

    audio_val = val['val_audio_data']
    pic_val = val['val_pic_data']
    labels_val = val['val_label_data']

    audio_test = test['test_audio_data']
    pic_test = test['test_pic_data']
    labels_test = test['test_label_data']

    audio_train, audio_val, audio_test = normalize_data(audio_train, audio_val, audio_test)

    print("shapes of train is {}, {} and shape of label is {}".format(audio_train.shape,
                                                                      pic_train.shape,
                                                                      labels_train.shape))
    print("shapes of val is {}, {} and shape of label is {}".format(audio_val.shape,
                                                                    pic_val.shape,
                                                                    labels_val.shape))
    print("shapes of test is {}, {} and shape of label is {}".format(audio_test.shape,
                                                                     pic_test.shape,
                                                                     labels_test.shape))

    train_data = (audio_train, pic_train, labels_train)
    val_data = (audio_val, pic_val, labels_val)
    test_data = (audio_test, pic_test, labels_test)

    return train_data, val_data, test_data


FINALl_EMOTIONS = {'sad': 0, 'neu': 1, 'hap': 2, 'ang': 3, 'fru': 4, 'exc': 5, 'oth': 6}


def load_subset_labels_data(config, labels=('sad', 'neu', 'hap', 'ang')):
    """
    Loads train and test data for video models
    :param labels: subset of all labels
    :param config: configuration file
    :return: train and test data
    """
    lbs = []
    for lb in labels:
        lbs.append(FINALl_EMOTIONS[lb])

    train_pkl = config['data']['train_pkl']
    val_pkl = config['data']['val_pkl']
    test_pkl = config['data']['test_pkl']

    train = load_pickle(train_pkl)
    val = load_pickle(val_pkl)
    test = load_pickle(test_pkl)
    # loading datasets
    audio_train = train['train_audio_data']
    pic_train = train['train_pic_data']
    labels_train = train['train_label_data']

    audio_val = val['val_audio_data']
    pic_val = val['val_pic_data']
    labels_val = val['val_label_data']

    audio_test = test['test_audio_data']
    pic_test = test['test_pic_data']
    labels_test = test['test_label_data']

    sub_tr_idx = np.fromiter((i for i, x in enumerate(labels_train) if x in lbs), dtype=labels_train.dtype)
    sub_val_idx = np.fromiter((i for i, x in enumerate(labels_val) if x in lbs), dtype=labels_val.dtype)
    sub_te_idx = np.fromiter((i for i, x in enumerate(labels_test) if x in lbs), dtype=labels_test.dtype)

    audio_train = audio_train[sub_tr_idx]
    pic_train = pic_train[sub_tr_idx]
    labels_train = labels_train[sub_tr_idx]

    audio_val = audio_val[sub_val_idx]
    pic_val = pic_val[sub_val_idx]
    labels_val = labels_val[sub_val_idx]

    audio_test = audio_test[sub_te_idx]
    pic_test = pic_test[sub_te_idx]
    labels_test = labels_test[sub_te_idx]

    audio_train, audio_val, audio_test = normalize_data(audio_train, audio_val, audio_test)

    print("shapes of train is {}, {} and shape of label is {}".format(audio_train.shape,
                                                                      pic_train.shape,
                                                                      labels_train.shape))
    print("shapes of val is {}, {} and shape of label is {}".format(audio_val.shape,
                                                                    pic_val.shape,
                                                                    labels_val.shape))
    print("shapes of test is {}, {} and shape of label is {}".format(audio_test.shape,
                                                                     pic_test.shape,
                                                                     labels_test.shape))

    train_data = (audio_train, pic_train, labels_train)
    val_data = (audio_val, pic_val, labels_val)
    test_data = (audio_test, pic_test, labels_test)

    return train_data, val_data, test_data
