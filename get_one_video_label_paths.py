from src.preprocessing import *
from src.extracting_video_paths import *


path_cfg = 'configs/config_paths.yml'
cfg = load_cfg(path_cfg)
logs_path = cfg['logs']['logs_path']
preprocessed_data_path = cfg['data']['preprocessed_data_path']
if not os.path.exists(preprocessed_data_path):
    os.makedirs(preprocessed_data_path)

iemocap_data_path = cfg['data']['iemocap_data_path']

enterfece_data_path = cfg['data']['enterfece_data_path']
ravdess_data_path = cfg['data']['ravdess_data_path']
iemocap_cliped_video_path = cfg['data']['iemocap_cliped_video_path']

filtered_save_path = cfg['data']['filtered_save_path']
if not os.path.exists(filtered_save_path):
    os.makedirs(filtered_save_path)

get_enterface_paths(enterfece_data_path, preprocessed_data_path)
get_ravdess_paths(ravdess_data_path, preprocessed_data_path)
iemocap_divide_videos_to_clips(iemocap_data_path, preprocessed_data_path, iemocap_cliped_video_path)

get_final_paths(preprocessed_data_path, filtered_save_path)
