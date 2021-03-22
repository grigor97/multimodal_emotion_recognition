import os
# import sys
import csv
import re
# import math
import pandas as pd
# import pdb

# import librosa
# import soundfile as sf
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# import matplotlib.style as ms
# from tqdm import tqdm
# import pickle
#
# import IPython.display
# import librosa.display
# ms.use('seaborn-muted')
# %matplotlib inline

# iemocap_full_release_path = "/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/IEMOCAP_full_release/"
# iemocap_pre_processed_data_path = "/Users/grigorkeropyan/Desktop/YSU_thesis/small_data/pre_processed_data/iemocap/"
iemocap_full_release_path = "/home/student/keropyan/data/IEMOCAP_full_release/"
iemocap_pre_processed_data_path = "/home/student/keropyan/data/pre_processed_data/iemocap/"


info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)

start_times, end_times, wav_file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []

for sess in range(1, 2):
    emo_evaluation_dir = iemocap_full_release_path + '/Session{}/dialog/EmoEvaluation/'.format(sess)
    evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
    for file in evaluation_files:
        x = re.search("^Ses.*", file)
        if x == None:
            continue
        with open(emo_evaluation_dir + file) as f:
            content = f.read()
        info_lines = re.findall(info_line, content)
        for line in info_lines[1:]:  # the first line is a header
            #pdb.set_trace()
            start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
            start_time, end_time = start_end_time[1:-1].split('-')
            val, act, dom = val_act_dom[1:-1].split(',')
            val, act, dom = float(val), float(act), float(dom)
            start_time, end_time = float(start_time), float(end_time)
            start_times.append(start_time)
            end_times.append(end_time)
            wav_file_names.append(wav_file_name)
            emotions.append(emotion)
            vals.append(val)
            acts.append(act)
            doms.append(dom)


df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom'])

df_iemocap['start_time'] = start_times
df_iemocap['end_time'] = end_times
df_iemocap['wav_file'] = wav_file_names
df_iemocap['emotion'] = emotions
df_iemocap['val'] = vals
df_iemocap['act'] = acts
df_iemocap['dom'] = doms

df_iemocap.to_csv(iemocap_pre_processed_data_path + '/df_iemocap_name_label.csv', index=False)