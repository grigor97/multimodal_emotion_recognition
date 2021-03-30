from utils.utils import *
from preprocessing import *
from extracting_video_paths import *


# TODO create folders if do not exist
path_cfg = 'configs/config_paths.yml'
cfg = load_cfg(path_cfg)
logs_path = cfg['logs']['logs_path']
preprocessed_data_path = cfg['data']['preprocessed_data_path']
iemocap_data_path = cfg['data']['iemocap_data_path']
enterfece_data_path = cfg['data']['enterfece_data_path']
ravdess_data_path = cfg['data']['ravdess_data_path']
iemocap_cliped_video_path = cfg['data']['iemocap_cliped_video_path']
filtered_save_path = cfg['data']['filtered_save_path']


save_test_data_path = cfg['data']['save_test_data_path']
test_paths = cfg['data']['test_paths']

save_train_data_path = cfg['data']['save_train_data_path']
save_train_name = cfg['data']['save_train_name']
save_test_name = cfg['data']['save_test_name']
train_paths = cfg['data']['train_paths']

get_enterface_paths(enterfece_data_path, preprocessed_data_path)
get_ravdess_paths(ravdess_data_path, preprocessed_data_path)
iemocap_divide_videos_to_clips(iemocap_data_path, preprocessed_data_path, iemocap_cliped_video_path)
get_final_paths(preprocessed_data_path, filtered_save_path)


test_df_paths = pd.read_csv(test_paths)
prepare_whole_data(test_df_paths, save_test_data_path, save_test_name)

train_df_paths = pd.read_csv(train_paths)
prepare_whole_data(train_df_paths, save_train_data_path, save_train_name)
