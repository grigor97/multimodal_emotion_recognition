import numpy as np
from PIL import Image
import pandas as pd
import glob
from utils.utils import *
import matplotlib.pyplot as plt


def load_images_for_one_video(pics_path):
    pcs = []

    for pic in glob.glob(pics_path + '*.jpg'):
        img = Image.open(pic)
        img = img.convert('RGB')

        tr_img = img.resize((300, 200))
        tr_img = np.array(tr_img)
        pcs.append(tr_img)

        del img
        del tr_img

    pcs = np.asarray(pcs)

    return pcs


final_emos = {'sad': 0, 'neu': 1, 'hap': 2, 'ang': 3, 'fru': 4, 'exc': 5, 'oth': 6}


# TODO fix paths
def get_pickle_file_from_all_pics(cfg):
    test_path = cfg['data']['save_test_data_path'] + cfg['data']['save_test_name']
    train_path = cfg['data']['save_train_data_path'] + cfg['data']['save_train_name']

    test = pd.read_csv(test_path)
    train = pd.read_csv(train_path)
    train.dropna(inplace=True)
    test.dropna(inplace=True)

    test_pictures = []
    test_labels = []
    for i, row in test.iterrows():
        imgs = load_images_for_one_video(row[0].replace('/home/student/keropyan', '..'))
        test_pictures.append(imgs)
        test_labels.append(final_emos[row[2]])

    train_labels = []
    train_pictures = []
    for i, row in train.iterrows():
        imgs = load_images_for_one_video(row[0].replace('/home/student/keropyan', '..'))
        train_pictures.append(imgs)
        train_labels.append(row[2])

    test_pictures = np.asarray(test_pictures)
    train_pictures = np.asarray(train_pictures)
    test_labels = np.asarray(test_labels)
    train_labels = np.asarray(train_labels)

    data = {"train_pics": train_pictures, "train_emotion": train_labels,
            "test_pics": test_pictures, "test_emotion": test_labels}
    pictures_with_emotions_pickle = cfg['data']['pictures_with_emotions_pickle']
    save_pickle(pictures_with_emotions_pickle, data)


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
