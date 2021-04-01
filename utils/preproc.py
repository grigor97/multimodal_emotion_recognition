import numpy as np
import pandas as pd

import librosa

import glob
from PIL import Image
import face_recognition

from utils.utils import *

final_emos = {'sad': 0, 'neu': 1, 'hap': 2, 'ang': 3, 'fru': 4, 'exc': 5, 'oth': 6}


def get_pickle_file_from_all_pics(cfg):
    """
    Creates pickle file for all the train and test data
    :param cfg: config file for paths
    :return: Nothing, just saves data in pickle file
    """
    test_path = cfg['data']['save_test_data_path'] + cfg['data']['save_test_name']
    train_path = cfg['data']['save_train_data_path'] + cfg['data']['save_train_name']

    test = pd.read_csv(test_path)
    train = pd.read_csv(train_path)
    train.dropna(inplace=True)
    test.dropna(inplace=True)

    test_audio_data, test_pic_data, test_label_data = get_features_for_df(test)
    train_audio_data, train_pic_data, train_label_data = get_features_for_df(test)

    data = {'train_audio_data': train_audio_data,
            'train_pic_data': train_pic_data,
            'train_label_data': train_label_data,
            'test_audio_data': test_audio_data,
            'test_pic_data': test_pic_data,
            'test_label_data': test_label_data
            }

    data_pickle = cfg['data']['train_test_data.pkl']
    save_pickle(data_pickle, data)


# TODO fix paths
def get_features_for_df(df):
    audio_data, pic_data, label_data = [], [], []
    for i, row in df.iterrows():
        pics_path = row[0].replace('/home/student/keropyan', '..')
        audio_path = row[1].replace('/home/student/keropyan', '..').replace('.npy', '.wav')
        label = final_emos[row[2]]
        audio_fs, pics_fs, labels = get_features_for_one_video(pics_path, audio_path, label)
        audio_data.extend(audio_fs)
        pic_data.extend(pics_fs)
        label_data.extend(labels)

    return np.asarray(audio_data), np.asarray(pic_data), np.asarray(label_data)


def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectrogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


def get_audio_features(path):
    # duration and offset are used to take care of the no
    # audio in start and the ending of each audio files as seen above.
    # TODO fix this part of duration and offset
    # data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    data, sample_rate = librosa.load(path)

    # without augmentation
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data, sample_rate)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))  # stacking vertically

    # data with shift
    shft_data = shift(data)
    res4 = extract_features(shft_data, sample_rate)
    result = np.vstack((result, res4))

    return result


def get_features_for_one_video(pics_path, wav_path, label):
    faces = load_faces_for_one_video(pics_path)
    audio_features = get_audio_features(wav_path)

    audio_fs = [audio_features[0], audio_features[1], audio_features[2], audio_features[3]]
    pics_fs = [faces, faces, faces, faces]
    labels = [label, label, label, label]

    return audio_fs, pics_fs, labels


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


def load_faces_for_one_video(pics_path):
    """
    Loads all the faces for one video
    :param pics_path: path to the pictures corresponding to one video
    :return: numpy array for faces for one video (20 images)
    """
    pcs = []

    for pic in glob.glob(pics_path + '*.jpg'):
        img = face_extraction(pic)
        pcs.append(img)

    pcs = np.asarray(pcs)
    return pcs
