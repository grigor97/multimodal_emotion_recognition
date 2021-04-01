import numpy as np
import pandas as pd

import glob
from PIL import Image
import face_recognition

from utils.utils import *

final_emos = {'sad': 0, 'neu': 1, 'hap': 2, 'ang': 3, 'fru': 4, 'exc': 5, 'oth': 6}


# TODO fix paths
def get_pickle_file_from_all_pics(cfg):
    """
    Creates pickle file for all the dataset images and labels
    :param cfg: config file for paths
    :return: Nothing, just saves data in pickle file
    """
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

        del imgs

    train_labels = []
    train_pictures = []
    for i, row in train.iterrows():
        imgs = load_images_for_one_video(row[0].replace('/home/student/keropyan', '..'))
        train_pictures.append(imgs)
        train_labels.append(row[2])

        del imgs

    test_pictures = np.asarray(test_pictures)
    train_pictures = np.asarray(train_pictures)
    test_labels = np.asarray(test_labels)
    train_labels = np.asarray(train_labels)

    data = {"train_pics": train_pictures, "train_emotion": train_labels,
            "test_pics": test_pictures, "test_emotion": test_labels}
    pictures_with_emotions_pickle = cfg['data']['pictures_with_emotions_pickle']
    save_pickle(pictures_with_emotions_pickle, data)


def face_extraction(pic_path, face_size=(50, 50)):
    """
    Extracting and return a dace from an image. This function uses Dlib cpp library for cpu case, it also
    deep neural network cnn which performance is better but it requires gpu!
    :param pic_path: path of the picture from which face should be extracted
    :param face_size: size of the face after extracting
    :return: returns a numpy array, only the face part of the image if at least one is
    detected otherwise returns whole image
    """
    image = face_recognition.load_image_file(pic_path)
    face_locs = face_recognition.face_locations(image)

    if len(face_locs) > 0:
        top, right, bottom, left = face_locs[0]
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
    else:
        pil_image = Image.fromarray(image)

    pil_image = pil_image.resize(face_size)
    return np.array(pil_image)


def load_images_for_one_video(pics_path):
    """
    Loads all the images for one video
    :param pics_path: path to the pictures corresponding to one video
    :return: numpy array for images for one video (20 images)
    """
    pcs = []

    for pic in glob.glob(pics_path + '*.jpg'):
        img = face_extraction(pic)
        pcs.append(img)

    pcs = np.asarray(pcs)
    return pcs
