import cv2
import math
import numpy as np
import pandas as pd
import librosa
from moviepy.editor import *

ONE_CLIP_LENGTH = 3  # each video length is 3 seconds
OVERLAP = 1  # when we have a long video which  corresponds to one
# emotion we divide it into parts where overlap is 1 seconds (4 seconds video --> 0-3 and 2-4)
DISREGARD_LENGTH = 0.5  # if video length is less than DISREGARD_LENGTH then we do not consider this video


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


def get_features(path):
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


def clip_video(video_path, start_time, end_time, save_path):
    video = VideoFileClip(video_path)
    if end_time > video.duration:
        end_time = video.duration

    video = video.subclip(start_time, end_time)
    video.write_videofile(
        save_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True)

    del video
    return save_path


def clip_audio(audio_path, start_time, end_time, save_path):
    audio = AudioFileClip(audio_path)
    if end_time > audio.duration:
        end_time = audio.duration

    audio = audio.subclip(start_time, end_time)
    audio.write_audiofile(
        save_path)

    del audio
    return save_path


def iemocap_extract_video_images_and_audio_features(vid_path, st, et, all_data_path, nth_sub_video):
    path_to_audio = vid_path.split('.')[0] + '.wav'
    vid_n = os.path.basename(vid_path)
    vid_name = vid_n.split(".")[0]
    num_images = 19

    save_folder = all_data_path + vid_name + '_' + str(nth_sub_video) + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    path_to_clip = clip_video(vid_path, st, et, save_folder + vid_name + '_' + str(nth_sub_video) + '.mp4')

    path_to_audio = clip_audio(path_to_audio, st, et, save_folder + vid_name + '_' + str(nth_sub_video) + '.wav')

    audio_features = get_features(path_to_audio)
    audio_features_path = save_folder + vid_name + '_' + str(nth_sub_video) + '_features.npy'
    np.save(audio_features_path, audio_features)

    cap = cv2.VideoCapture(path_to_clip)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    interval = length // num_images
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
        if (frame_id <= (length - length % num_images)) and (frame_id % math.floor(interval) == 0):

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
                # Get right part of image
            print("After", frame.shape)

            if not os.path.exists(pic_path):
                os.makedirs(pic_path)
            cv2.imwrite(filename, frame)

    cap.release()
    print("Done!")
    return pic_path, audio_features_path


def other_extract_video_images_and_audio_features(vid_path, st, et, all_data_path, nth_sub_video):
    vid_n = os.path.basename(vid_path)
    vid_name = vid_n.split(".")[0]
    num_images = 19

    save_folder = all_data_path + vid_name + '_' + str(nth_sub_video) + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    path_to_clip = clip_video(vid_path, st, et, save_folder + vid_name + '_' + str(nth_sub_video) + '.mp4')

    audio_clip = AudioFileClip(path_to_clip)
    audio_path = save_folder + vid_name + '_' + str(nth_sub_video) + '.wav'
    audio_clip.write_audiofile(audio_path)

    audio_features = get_features(audio_path)
    audio_features_path = save_folder + vid_name + '_' + str(nth_sub_video) + '_features.npy'
    np.save(audio_features_path, audio_features)

    cap = cv2.VideoCapture(path_to_clip)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    interval = length // num_images
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
        if (frame_id <= (length - length % num_images)) and (frame_id % math.floor(interval) == 0):

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
                # Get right part of image
            print("After", frame.shape)

            if not os.path.exists(pic_path):
                os.makedirs(pic_path)
            cv2.imwrite(filename, frame)

    cap.release()
    print("Done!")
    return pic_path, audio_features_path


def extract_video_images_and_audio_features(vid_path, st, et, all_data_path, nth_sub_video):
    if 'iemocap' in vid_path:
        return iemocap_extract_video_images_and_audio_features(vid_path,
                                                               st,
                                                               et,
                                                               all_data_path,
                                                               nth_sub_video)
    else:
        return other_extract_video_images_and_audio_features(vid_path,
                                                               st,
                                                               et,
                                                               all_data_path,
                                                               nth_sub_video)


def prepare_one_video(video_path, save_data_path):
    video = VideoFileClip(video_path)
    video_length = video.duration

    paths = []
    cnt = 0
    start = 0
    while start < video_length:
        end = start + 3
        if end > video_length:
            end = video_length
        if start > 0 and end - start < 2 * OVERLAP:
            break

        pth = extract_video_images_and_audio_features(video_path, start, end, save_data_path, cnt)
        paths.append(pth)

        start = start + ONE_CLIP_LENGTH - OVERLAP
        cnt += 1

    return paths  # pictures paths and npy file paths which is audio features


def prepare_whole_data(data_paths, save_data_path, save_name):
    pic_paths, npy_paths, emotions = [], [], []

    for i, row in data_paths.iterrows():
        paths = prepare_one_video(row[0], save_data_path)
        for path in paths:
            pic_paths.append(path[0])
            npy_paths.append(path[1])
            emotions.append(row[1])

    df = pd.DataFrame(columns=['pic_paths', 'npy_paths', 'emotion'])

    df['pic_paths'] = pic_paths
    df['npy_paths'] = npy_paths
    df['emotion'] = emotions
    df.to_csv(save_data_path + save_name, index=False)
    return df
