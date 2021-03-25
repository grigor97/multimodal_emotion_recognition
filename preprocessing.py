import cv2
import math
import numpy as np
from moviepy.editor import *


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


def extract_video_images(vid_path, st, et, all_data_path):
    vid_n = os.path.basename(vid_path)
    vid_name = vid_n.split(".")[0]
    num_images = 19

    save_folder = all_data_path + vid_name + '/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    path_to_clip = clip_video(vid_path, st, et, save_folder + vid_n)

    cap = cv2.VideoCapture(path_to_clip)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    interval = length // num_images
    frame_rate = cap.get(5)  # frame rate
    print(frame_rate)
    x = 1
    while cap.isOpened():
        frame_id = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if not ret:
            break
        if length % num_images == 0:
            length -= 1
        if (frame_id <= (length - length % num_images)) and (frame_id % math.floor(interval) == 0):
            pic_path = all_data_path + vid_name + '/pics/'
            filename = pic_path + str(vid_name) + "_" + str(int(x)) + ".jpg"
            x += 1
            print("Frame shape Before resize", frame.shape)
            m_f_i = vid_name.split("_")
            m_f_l = m_f_i[0][-1]
            m_f_r = m_f_i[2][0]
            y1 = frame.shape[0]
            w1 = frame.shape[1]
            new_x = np.int(w1/2)
            yy = np.int(y1/4)
            if m_f_r == m_f_l:
                # Get left part of image
                frame = frame[yy: 3*yy, 0:new_x, :]
            else:
                frame = frame[yy: 3*yy, new_x:w1, :]
                # Get right part of image
            print("After", frame.shape)

            if not os.path.exists(pic_path):
                os.makedirs(pic_path)
            cv2.imwrite(filename, frame)

    cap.release()
    print("Done!")
