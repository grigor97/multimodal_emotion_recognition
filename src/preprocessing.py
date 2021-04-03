import math
import numpy as np
import pandas as pd

import librosa

import glob
from PIL import Image
import face_recognition
import cv2
from moviepy.editor import *

from utils.utils import *

FINALl_EMOTIONS = {'sad': 0, 'neu': 1, 'hap': 2, 'ang': 3, 'fru': 4, 'exc': 5, 'oth': 6}


def get_pickle_file_from_all_pics_and_audios(cfg):
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

    test_data = {
            'test_audio_data': test_audio_data,
            'test_pic_data': test_pic_data,
            'test_label_data': test_label_data
            }

    test_pickle = cfg['data']['test_pkl']
    save_pickle(test_pickle, test_data)

    train_audio_data, train_pic_data, train_label_data = get_features_for_df(train)
    train_data = {'train_audio_data': train_audio_data,
                  'train_pic_data': train_pic_data,
                  'train_label_data': train_label_data,
                  }

    train_pickle = cfg['data']['train_pkl']
    save_pickle(train_pickle, train_data)


def get_features_for_df(df):
    audio_data, pic_data, label_data = [], [], []
    for i, row in df.iterrows():
        pics_path = row[0]
        audio_path = row[1]
        label = FINALl_EMOTIONS[row[2]]
        audio_fs, pics_fs, labels = get_features_for_one_video(pics_path, audio_path, label)
        audio_data.extend(audio_fs)
        pic_data.extend(pics_fs)
        label_data.extend(labels)

    return np.asarray(audio_data), np.asarray(pic_data), np.asarray(label_data)

# def get_features_for_df(df):
#     audio_data, pic_data, label_data = [], [], []
#     for i, row in df.iterrows():
#         pics_path = row[0].replace('/home/student/keropyan', '..')
#         # extracting wav file path from npy file path
#         audio_path = row[1].replace('/home/student/keropyan', '..')[:-13] + '.wav'
#         label = FINALl_EMOTIONS[row[2]]
#         audio_fs, pics_fs, labels = get_features_for_one_video(pics_path, audio_path, label)
#         audio_data.extend(audio_fs)
#         pic_data.extend(pics_fs)
#         label_data.extend(labels)
#
#     return np.asarray(audio_data), np.asarray(pic_data), np.asarray(label_data)


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
    :return: returns a numpy array, only the grayscale face part of the image if at least one is
    detected otherwise returns whole grayscale image
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
    # make it grayscale
    pil_image = pil_image.convert('L')
    return np.array(pil_image)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


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


ONE_CLIP_LENGTH = 3  # each video length is 3 seconds
OVERLAP = 1  # when we have a long video which  corresponds to one
# emotion we divide it into parts where overlap is 1 seconds (4 seconds video --> 0-3 and 2-4)
DISREGARD_LENGTH = 0.5  # if video length is less than DISREGARD_LENGTH then we do not consider this video
# PIC_DIMS = (600, 360)
PIC_DIMS = (360, 240)


def clip_video(video_path, audio_save_path, start_time, end_time, save_path):
    video = VideoFileClip(video_path)
    if end_time > video.duration:
        end_time = video.duration

    sub_video = video.subclip(start_time, end_time)
    sub_video.write_videofile(
        save_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True)

    sub_video.audio.write_videofile(audio_save_path)

    # del video.reader
    # del sub_video.reader
    del video
    del sub_video

    return save_path


def clip_audio(audio_path, start_time, end_time, save_path):
    audio = AudioFileClip(audio_path)
    if end_time > audio.duration:
        end_time = audio.duration

    sub_audio = audio.subclip(start_time, end_time)
    sub_audio.write_audiofile(
        save_path)

    # del audio.reader
    # del sub_audio.reader
    del audio
    del sub_audio

    return save_path


def iemocap_extract_video_images_and_audio_features(vid_path, st, et, all_data_path, nth_sub_video):
    path_to_audio = vid_path.split('.')[0] + '.wav'
    vid_n = os.path.basename(vid_path)
    vid_name = vid_n.split(".")[0]
    num_images = 19

    save_folder = all_data_path + vid_name + '_' + str(nth_sub_video) + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    audio_save_path = save_folder + vid_name + '_' + str(nth_sub_video) + '.wav'
    path_to_clip = clip_video(vid_path, audio_save_path, st, et, save_folder + vid_name + '_' + str(nth_sub_video) + '.mp4')

    # path_to_audio = clip_audio(path_to_audio, st, et, save_folder + vid_name + '_' + str(nth_sub_video) + '.wav')

    # audio_features = get_audio_features(path_to_audio)
    # audio_features_path = save_folder + vid_name + '_' + str(nth_sub_video) + '_features.npy'
    # np.save(audio_features_path, audio_features)

    cap = cv2.VideoCapture(path_to_clip)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    interval = math.floor(length // num_images)
    if interval < 1:
        return None, None
    frame_rate = cap.get(5)  # frame rate
    print(frame_rate)

    x = 1
    pic_path = save_folder + 'pics/'
    while cap.isOpened():
        frame_id = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if not ret:
            break
        if length % num_images == 0:
            length -= 1
        if (frame_id <= (length - length % num_images)) and (frame_id % interval == 0):

            filename = pic_path + str(vid_name) + '_' + str(nth_sub_video) + "_" + str(int(x)) + ".jpg"
            x += 1
            print("Frame shape Before resize", frame.shape)
            m_f_i = vid_name.split("_")
            m_f_l = m_f_i[0][-1]
            m_f_r = m_f_i[2][0]
            y1 = frame.shape[0]
            w1 = frame.shape[1]
            new_x = np.int(w1 / 2)
            yy = np.int(y1 / 4)
            if m_f_r == m_f_l:
                # Get left part of image
                frame = frame[yy: 3 * yy, 0:new_x, :]
            else:
                frame = frame[yy: 3 * yy, new_x:w1, :]

                # frame = cv2.resize(frame, PIC_DIMS, interpolation=cv2.INTER_LINEAR)
                # Get right part of image
            print("After", frame.shape)

            if not os.path.exists(pic_path):
                os.makedirs(pic_path)
            cv2.imwrite(filename, frame)

    cap.release()
    print("Done!")
    return pic_path, path_to_audio


def other_extract_video_images_and_audio_features(vid_path, st, et, all_data_path, nth_sub_video):
    vid_n = os.path.basename(vid_path)
    vid_name = vid_n.split(".")[0]
    num_images = 19

    save_folder = all_data_path + vid_name + '_' + str(nth_sub_video) + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    audio_path = save_folder + vid_name + '_' + str(nth_sub_video) + '.wav'
    path_to_clip = clip_video(vid_path, audio_path, st, et, save_folder + vid_name + '_' + str(nth_sub_video) + '.mp4')

    # audio_clip = AudioFileClip(path_to_clip)
    # audio_clip.write_audiofile(audio_path)

    # del audio_clip.reader
    # del audio_clip

    # audio_features = get_audio_features(audio_path)
    # audio_features_path = save_folder + vid_name + '_' + str(nth_sub_video) + '_features.npy'
    # np.save(audio_features_path, audio_features)

    cap = cv2.VideoCapture(path_to_clip)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    interval = math.floor(length // num_images)
    if interval < 1:
        return None, None
    frame_rate = cap.get(5)  # frame rate
    print(frame_rate)

    x = 1
    pic_path = save_folder + 'pics/'
    while cap.isOpened():
        frame_id = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if not ret:
            break
        if length % num_images == 0:
            length -= 1
        if (frame_id <= (length - length % num_images)) and (frame_id % interval == 0):

            filename = pic_path + str(vid_name) + '_' + str(nth_sub_video) + "_" + str(int(x)) + ".jpg"
            x += 1

            if not os.path.exists(pic_path):
                os.makedirs(pic_path)
            cv2.imwrite(filename, frame)

    cap.release()
    print("Done!")
    return pic_path, audio_path


def prepare_one_video(video_path, save_data_path):
    video = VideoFileClip(video_path)
    video_length = video.duration

    paths = []
    cnt = 0
    start = 0

    if 'iemocap' not in video_path:
        pth = other_extract_video_images_and_audio_features(video_path, 0.6, 3.6, save_data_path, 0)
        if pth[0] is not None:
            paths.append(pth)

        return paths

    while start < video_length:
        end = start + 3
        if end > video_length:
            end = video_length
        if start > 0 and end - start < 2 * OVERLAP:
            break

        pth = iemocap_extract_video_images_and_audio_features(video_path, start, end, save_data_path, cnt)
        if pth[0] is not None:
            paths.append(pth)

        start = start + ONE_CLIP_LENGTH - OVERLAP
        cnt += 1

    # del video.reader
    del video

    return paths  # pictures paths and npy file paths which is audio features


def prepare_whole_data(data_paths, save_data_path, save_name):
    pic_paths, wav_paths, emotions = [], [], []

    for i, row in data_paths.iterrows():
        paths = prepare_one_video(row[0], save_data_path)
        for path in paths:
            pic_paths.append(path[0])
            wav_paths.append(path[1])
            emotions.append(row[1])

    df = pd.DataFrame(columns=['pic_paths', 'npy_paths', 'emotion'])

    df['pic_paths'] = pic_paths
    df['wav_paths'] = wav_paths
    df['emotion'] = emotions
    df.to_csv(save_data_path + save_name, index=False)
    return df
