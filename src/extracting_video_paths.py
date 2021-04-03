import re
from glob import glob
from moviepy.editor import *
import pandas as pd


ravdess_emo_conv_dict = {
    '06': 'fea',
    '08': 'sur',
    '04': 'sad',
    '03': 'hap',
    '05': 'ang',
    '07': 'dis',
    '01': 'neu',
    '02': 'calm'
}


def get_ravdess_paths(ravdess_data_path, pre_processed_data_path):
    emotions, actors, video_paths, sexes = [], [], [], []
    for actor in os.listdir(ravdess_data_path):
        if actor == '.DS_Store':
            continue
        act_path = ravdess_data_path + '/' + actor

        ps = glob(act_path + '/' + '*.mp4')
        for p in ps:
            vals = p.split('/')[-1].split('.')[0].split('-')
            # filtering only videos with audio
            if vals[0] != '01':
                continue

            ac = int(vals[-1])
            actors.append(ac)
            if ac % 2 == 0:
                sexes.append('F')
            else:
                sexes.append('M')

            video_paths.append(p)
            emotions.append(ravdess_emo_conv_dict[vals[2]])

    df_ravdess = pd.DataFrame(columns=['file_path', 'emotion', 'actor', 'sex'])

    df_ravdess['file_path'] = video_paths
    df_ravdess['emotion'] = emotions
    df_ravdess['actor'] = actors
    df_ravdess['sex'] = sexes

    df_ravdess.to_csv(pre_processed_data_path + '/df_ravdess.csv', index=False)


enterface_emo_conv_dict = {
    'fear': 'fea',
    'surprise': 'sur',
    'sadness': 'sad',
    'happiness': 'hap',
    'anger': 'ang',
    'disgust': 'dis',
}


def get_enterface_paths(enterfece_data_path, pre_processed_data_path):
    video_paths, emotions, subjects = [], [], []
    for subject in os.listdir(enterfece_data_path):
        if subject == '.DS_Store':
            continue
        sub_path = enterfece_data_path + subject
        for emotion in os.listdir(sub_path):
            if emotion == '.DS_Store':
                continue
            emo_path = sub_path + '/' + emotion

            for sent in os.listdir(emo_path):
                if sent == '.DS_Store':
                    continue

                sent_path = emo_path + '/' + sent
                paths = glob(sent_path + '/' + '*.avi')
                for p in paths:
                    video_paths.append(p)
                    emotions.append(enterface_emo_conv_dict[emotion])
                    subjects.append(int(subject.split(' ')[1]))

    df_enterface = pd.DataFrame(columns=['file_path', 'emotion', 'subject'])
    df_enterface['file_path'] = video_paths
    df_enterface['emotion'] = emotions
    df_enterface['subject'] = subjects
    df_enterface.to_csv(pre_processed_data_path + '/df_enterface.csv', index=False)


def iemocap_divide_videos_to_clips(data_path, pre_processed_data_path, video_save_path ):
    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
    start_times, end_times, file_paths, emotions, sessions, sexes = [], [], [], [], [], []
    # for x in range(1):
    for x in range(5):
        sess_name = "Session" + str(x + 1)
        print('processing in session ', sess_name)
        path_video = data_path + sess_name + "/dialog/avi/DivX/"
        path_label = data_path + sess_name + "/dialog/EmoEvaluation/"
        video_clip_path = video_save_path + sess_name + "/sentences_video_audio/"
        if not os.path.exists(video_clip_path):
            os.makedirs(video_clip_path)
        videos = glob(path_video + '*.avi')

        for video_name in videos:
            video_name = video_name.split("/")[-1]
            video_name_folder = video_clip_path + video_name.split(".")[0] + '/'
            if not os.path.exists(video_name_folder):
                os.makedirs(video_name_folder)
            with open(path_label + video_name.split(".")[0] + '.txt') as f:
                content = f.read()
            info_lines = re.findall(info_line, content)
            for line in info_lines[1:]:  # the first line is a header
                # print(path_label + video_name.split(".")[0] + '.txt')
                start_end_time, file_name, emotion, val_act_dom = line.strip().split('\t')
                start_time, end_time = start_end_time[1:-1].split('-')
                start_time, end_time = float(start_time), float(end_time)
                start_times.append(start_time)
                end_times.append(end_time)
                file_paths.append(video_name_folder + file_name + ".mp4")
                emotions.append(emotion)
                sessions.append(x + 1)

                sex = file_name.split('_')[-1][0]
                sexes.append(sex)

                video = VideoFileClip(path_video + video_name)
                if end_time > video.duration:
                    end_time = video.duration

                sub_video = video.subclip(start_time, end_time)
                sub_video.write_videofile(
                    video_name_folder + file_name + ".mp4",
                    audio=True
                    # codec='libx264',
                    # audio_codec='aac',
                    # temp_audiofile='temp-audio.m4a',
                    # remove_temp=True
                    )

                # TODO maybe remove
                # sub_video.audio.write_audiofile(video_name_folder + file_name + ".wav")

                # video.close()
                # del video.reader
                # del sub_video.reader
                # del video
                # del sub_video

    df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'file_path', 'emotion', 'session', 'sex'])

    df_iemocap['start_time'] = start_times
    df_iemocap['end_time'] = end_times
    df_iemocap['file_path'] = list(map(str, file_paths))
    df_iemocap['emotion'] = list(map(str, emotions))
    df_iemocap['session'] = list(map(int, sessions))
    df_iemocap['sex'] = list(map(str, sexes))
    df_iemocap.to_csv(pre_processed_data_path + '/df_iemocap.csv', index=False)


def get_final_paths(df_paths, save_path):
    # reading paths
    df_iemocap = pd.read_csv(df_paths + 'df_iemocap.csv')
    df_enterface = pd.read_csv(df_paths + 'df_enterface.csv')
    df_ravdess = pd.read_csv(df_paths + 'df_ravdess.csv')

    # filtering and combining some emotions!
    current_emos = ['sad', 'neu', 'hap', 'ang',
                    'fru', 'exc', 'oth', 'dis', 'fea', 'sur']

    final_emos = ['sad', 'neu', 'hap', 'ang', 'fru', 'exc', 'oth']
    df_iemocap = df_iemocap.loc[df_iemocap.emotion.isin(current_emos)]
    df_enterface = df_enterface.loc[df_enterface.emotion.isin(current_emos)]
    df_ravdess = df_ravdess.loc[df_ravdess.emotion.isin(current_emos)]

    df_iemocap['emotion'] = df_iemocap['emotion'].replace({'sur': 'oth', 'fea': 'oth', 'dis': 'oth'})
    df_enterface['emotion'] = df_enterface['emotion'].replace({'sur': 'oth', 'fea': 'oth', 'dis': 'oth'})
    df_ravdess['emotion'] = df_ravdess['emotion'].replace({'sur': 'oth', 'fea': 'oth', 'dis': 'oth'})

    df_iemocap.to_csv(save_path + 'df_iemocap.csv', index=False)
    df_ravdess.to_csv(save_path + 'df_ravdess.csv', index=False)
    df_enterface.to_csv(save_path + 'df_enterface.csv', index=False)

    iemocap_train = df_iemocap[df_iemocap['session'] != 4][['file_path', 'emotion']]
    iemocap_test = df_iemocap[df_iemocap['session'] == 4][['file_path', 'emotion']]

    enterface_train = df_enterface[df_enterface['subject'] <= 36][['file_path', 'emotion']]
    enterface_test = df_enterface[df_enterface['subject'] > 36][['file_path', 'emotion']]

    ravdess_train = df_ravdess[df_ravdess['actor'] <= 20][['file_path', 'emotion']]
    ravdess_test = df_ravdess[df_ravdess['actor'] > 20][['file_path', 'emotion']]

    train_paths = pd.concat([iemocap_train, enterface_train, ravdess_train])
    test_paths = pd.concat([iemocap_test, enterface_test, ravdess_test])

    train_paths.to_csv(save_path + 'train_paths.csv', index=False)
    test_paths.to_csv(save_path + 'test_paths.csv', index=False)