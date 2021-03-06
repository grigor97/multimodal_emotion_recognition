import numpy as np
import random
from utils.utils import *
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


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

    plt.plot(epochs, loss, 'g', label='train')
    plt.plot(epochs, val_loss, 'b', label='val')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper left')
    plt.legend()

    plt.savefig(save_path + "/loss.png")

    plt.clf()

    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']

    plt.plot(epochs, acc, 'g', label='train')
    plt.plot(epochs, val_acc, 'b', label='val')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='upper left')
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
    train_mean = audio_train.mean(axis=0)
    audio_train -= train_mean
    safe_max = np.abs(audio_train).max(axis=0)
    safe_max[safe_max == 0] = 1

    # np.save('logs/train_mean.npy', train_mean)
    # np.save('logs/safe_max.npy', safe_max)

    audio_val -= train_mean
    audio_test -= train_mean

    audio_train /= safe_max
    audio_val /= safe_max
    audio_test /= safe_max

    # print(train_mean)
    # print('safe max')
    # print(safe_max)

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


class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, tol=0.15, patience=4):
        super(CustomEarlyStopping, self).__init__()
        self.tol = tol
        self.patience = patience
        # self.best_weights = None
        # self.test_data = test_data

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # x, y = self.test_data
        # te_loss, te_acc = self.model.evaluate(x, y, verbose=0)
        # print('\nTesting loss: {}, acc: {}\n'.format(te_loss, te_acc))

        v_acc = logs.get('val_accuracy')
        t_acc = logs.get('accuracy')

        if t_acc - v_acc > self.tol:
            # if t_acc - v_acc > self.tol + 15:
            #     print('gap is too larger wait time is {}'.format(self.wait))
            #     self.wait += 100
            print('gap is larger wait time is {}'.format(self.wait))
            self.wait += 1
        else:
            self.wait = 0

        if self.wait > self.patience and epoch > 1:
            print('stopppppppinggggggggg')
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10, 10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)
    plt.clf()
